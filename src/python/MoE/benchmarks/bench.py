import copy
from threading import Thread
import time
import torch
import timeit
import optparse
from typing import Dict, List
import torch.nn.functional as F

from TensorTransferEngine.utils import HandshakeType, TransferProtocol
from TensorTransferEngine.TensorTransferEngine import TensorTransferEngineBackend, TensorTransferEngineConfig

from MoE.MoE import Qwen3MoEMLP
from MoE.DistMoEBlock import DistMoEBlock, DistMoEBlockConfig
from MoE.DistExpertsBlock import DistExpertsBlock

from utils.Benchmark import Benchmark
from utils.logging_config import get_logger

logger = get_logger("dist_moe.benchmarks.moe")


def parse_arguments():
    parser = optparse.OptionParser()
    # parser.add_option(
    #     "--schema",
    #     help="specify the schema of data transfers (push-push/pull-push)",
    #     nargs=1,
    #     type="str",
    #     default="pull-push",
    # )
    parser.add_option(
        "--EB",
        dest="EB",
        help="specify number of expert blocks",
        nargs=1,
        type="int",
        default=4,
    )
    parser.add_option(
        "--number_of_experts",
        dest="number_of_experts",
        help="specify number of expert in each of experts block",
        nargs=1,
        type="int",
        default=128,
    )
    parser.add_option(
        "--top_k",
        dest="top_k",
        help="specify top K value for a MoE layer",
        nargs=1,
        type="int",
        default=8,
    )
    parser.add_option(
        "--io_buffer_size",
        dest="io_buffer_size",
        help="specify the I/O buffers size (in tokens)",
        nargs=1,
        type="int",
        default=32000,
    )
    parser.add_option(
        "--warmup_batch_sizes",
        dest="warmup_batch_sizes",
        help="specify size of warm up batches (in tokens)",
        nargs=1,
        type="str",
        # default="4000,8000,16000,64000",
        default="8000",
    )
    parser.add_option(
        "--batch_sizes",
        dest="batch_sizes",
        help="specify size of perform batches (in tokens)",
        nargs=1,
        type="str",
        # default="4000,5000,6000,7000,8000,9000,10000,11000,12000,13000,14000,15000,16000,32000,64000,128000",
        default="8000,8000,8000",
    )
    parser.add_option(
        "--precision",
        dest="precision",
        help="specify MoE layer's tensors precision (int8/fp8/fp16/bf16/fp32)",
        nargs=1,
        type="str",
        default="fp16",
    )
    parser.add_option(
        "--host_address",
        dest="host_address",
        help="specify the host full address (with port)",
        nargs=1,
        type="str",
        default="localhost:10000",
    )
    parser.add_option(
        "--experts_address",
        dest="experts_address",
        help="specify the experts starting full address (with port)",
        nargs=1,
        type="str",
        default="localhost:20000",
    )
    parser.add_option(
        "--host_gpu_id",
        dest="host_gpu_id",
        help="specify host gpu id (from 0 to 7)",
        nargs=1,
        type="int",
        default=0,
    )
    parser.add_option(
        "--EB_gpu_ids",
        dest="EB_gpu_ids",
        help="specify expert block gpu ids (list of ints from 0 to 7)",
        nargs=1,
        type="str",
        default="1,2,3,4",
    )
    parser.add_option(
        "--sequence_length",
        dest="sequence_length",
        help="specify the sequence length of a MoE layer",
        nargs=1,
        type="int",
        default=1024,
    )
    parser.add_option(
        "--hidden_features",
        dest="hidden_features",
        help="specify the hidden features of a MoE layer",
        nargs=1,
        type="int",
        default=1024,
    )
    parser.add_option(
        "--intermediate_features",
        dest="intermediate_features",
        help="specify the intermediate features of a MoE layer (expert layer expansion)",
        nargs=1,
        type="int",
        default=4096,
    )

    (options, _) = parser.parse_args()
    batch_sizes = list(map(int, options.batch_sizes.split(",")))
    warmup_batch_sizes = list(map(int, options.warmup_batch_sizes.split(",")))

    return (options, batch_sizes, warmup_batch_sizes)


class Bench:
    def __init__(self, options, batch_sizes: List[int], warmup_batch_sizes: List[int]) -> None:
        assert torch.cuda.is_available()

        self.host_device = torch.device(f"cuda:{options.host_gpu_id}")
        self.expert_block_gpu_ids = list(map(int, options.EB_gpu_ids.split(",")))

        self.benchmarks: Dict[str, Benchmark] = {}
        self.benchmarks["e2e_moe"] = Benchmark()
        self.benchmarks["e2e_moe"].metadata["batch_size"] = []
        self.benchmarks["throughput"] = Benchmark()
        self.benchmarks["throughput"].metadata["batch_size"] = []
        self.benchmarks["throughput"].metadata["throughput"] = []
        self.benchmarks["gpu_memory"] = Benchmark()
        self.benchmarks["gpu_memory"].metadata["batch_size"] = []
        self.benchmarks["gpu_memory"].metadata["free_memory"] = []

        self.io_buffer_size = options.io_buffer_size
        self.number_of_expert_blocks = options.EB
        self.number_of_experts = options.number_of_experts
        self.top_k = options.top_k

        # self.schema = options.schema
        self.warmup_batch_sizes = warmup_batch_sizes
        self.batch_sizes = batch_sizes
        self.sequence_length = options.sequence_length
        self.hidden_features = options.hidden_features
        self.intermediate_features = options.intermediate_features

        self.dtype = None
        if options.precision == "fp8":
            self.dtype = torch.float8_e5m2
        elif options.precision == "fp16":
            self.dtype = torch.float16
        elif options.precision == "bf16":
            self.dtype = torch.bfloat16
        elif options.precision == "fp32":
            self.dtype = torch.float32
        elif options.precision == "int8":
            self.dtype = torch.int8
        assert self.dtype is not None

        self.host_address, self.host_port = options.host_address.split(":")
        self.experts_address, self.experts_port = options.experts_address.split(":")
        self.host_port = int(self.host_port)
        self.experts_port = int(self.experts_port)

        self.backends = ["nixl"]
        self.transfer_protocols = [TransferProtocol.RDMA]

    def setup_dist_moe(
        self,
        backend: TensorTransferEngineBackend,
        number_of_experts: int,
        number_of_expert_blocks: int,
        expert_block_devices: List[torch.device],
        top_k: int,
        transfer_protocol: TransferProtocol,
        host_ib_devices_names: str,
        expert_block_ib_devices_names: str,
    ):
        expert_block_ports_start_index = 20000
        assert self.host_port < expert_block_ports_start_index

        host_p2p_ports = [self.host_port + 1000 * (index + 1) for index in range(number_of_expert_blocks)]

        expert_block_hosts: List[str] = []
        expert_block_ports: List[int] = []
        expert_block_p2p_ports: List[int] = []
        expert_blocks_experts: List[List[int]] = []

        for expert_block_index in range(number_of_expert_blocks):
            expert_block_hosts.append(self.experts_address)
            expert_block_ports.append(expert_block_ports_start_index + 100 * expert_block_index)
            expert_block_p2p_ports.append(expert_block_ports_start_index + 100 * expert_block_index + 1000)
            expert_blocks_experts.append([])

        for expert_index in range(number_of_experts):
            for expert_block_experts in expert_blocks_experts:
                expert_block_experts.append(expert_index)

        assert self.dtype != None

        # Health check ports allocation (avoiding collisions with existing ports).
        host_health_check_ports = [self.host_port + 2000 * (index + 1) for index in range(number_of_expert_blocks)]
        remote_health_check_ports = [expert_block_ports_start_index + 100 * index + 2000 for index in range(number_of_expert_blocks)]

        dist_moe_config = DistMoEBlockConfig(
            backend=backend,
            expert_blocks=expert_blocks_experts,
            number_of_experts=number_of_experts,
            top_k=top_k,
            number_of_io_buffers=8,
            io_buffer_size=self.io_buffer_size,
            hidden_features=self.hidden_features,
            dtype=self.dtype,
            moe_device=self.host_device,
            eb_devices=expert_block_devices,
            forward_timeout=4000,
            host_health_check_ports=host_health_check_ports,
            remote_health_check_ports=remote_health_check_ports,
        )

        expert_blocks: List[DistExpertsBlock] = []
        expert_layers = [Qwen3MoEMLP(self.hidden_features, self.intermediate_features, self.dtype) for _ in range(number_of_experts)]
        for experts_block_index in range(number_of_expert_blocks):
            experts_block_layers = []
            for expert_index in expert_blocks_experts[experts_block_index]:
                experts_block_layers.append(copy.deepcopy(expert_layers[expert_index]))

            eb_tte_config = TensorTransferEngineConfig(
                host_address=expert_block_hosts[experts_block_index],
                host_port=expert_block_ports[experts_block_index],
                host_p2p_ports=[expert_block_p2p_ports[experts_block_index]],
                remote_addresses=["localhost"],
                remote_ports=[self.host_port],
                remote_p2p_ports=[host_p2p_ports[experts_block_index]],
                handshake_type=HandshakeType.NONE,
                transfer_protocol=TransferProtocol.RDMA,
                ib_device_names="",
                p2p_timeout_duration=3000,
                metadata_schema="http",
                metadata_host="localhost",
                metadata_port=8080,
                metadata_dir="metadata",
            )

            expert_blocks.append(DistExpertsBlock(experts_block_index, dist_moe_config, experts_block_layers, eb_tte_config))
            expert_blocks[-1].to(expert_block_devices[experts_block_index])

            t = Thread(target=expert_blocks[-1].main_loop, daemon=True)
            t.start()

            print("--------------------------------------------------------------------------------------")
            logger.info(
                f"The {backend.value} DistExpertsBlock executor has been opened (role: experts block; transfer protocol: {transfer_protocol.value}; experts: {expert_blocks_experts[experts_block_index]}; device: {expert_block_devices[experts_block_index]})..."
            )

        print("--------------------------------------------------------------------------------------")
        logger.info(
            f"The {backend.value} DistMoEBlock executor has been opened (role: host; transfer protocol: {transfer_protocol.value}; expert blocks: {number_of_expert_blocks})..."
        )
        logger.info(f"Working on {self.host_device} device.")

        dist_moe_tte_config = TensorTransferEngineConfig(
            host_address="localhost",
            host_port=self.host_port,
            host_p2p_ports=host_p2p_ports,
            remote_addresses=expert_block_hosts,
            remote_ports=expert_block_ports,
            remote_p2p_ports=expert_block_p2p_ports,
            handshake_type=HandshakeType.NONE,
            transfer_protocol=TransferProtocol.RDMA,
            ib_device_names="",
            p2p_timeout_duration=3000,
            metadata_schema="http",
            metadata_host="localhost",
            metadata_port=8080,
            metadata_dir="metadata",
        )

        moe_block = DistMoEBlock(dist_moe_config, dist_moe_tte_config)
        moe_block.to(self.host_device)

        return moe_block, expert_blocks

    def local_experts(
        self, moe_block: DistMoEBlock, expert_blocks: List[DistExpertsBlock], batch_size: int, batch: torch.Tensor, is_warmup: bool
    ):
        print("--------------------------------------------------------------------------------------")
        logger.info(f"The default MoE executor has been opened...")
        logger.info(f"Working on [{self.host_device}] device(s).")

        duration = timeit.default_timer()
        if not is_warmup:
            self.benchmarks["e2e_moe"].start()

        input_batch = batch.view(-1, moe_block.hidden_features)
        output_batch = torch.zeros(input_batch.shape, dtype=moe_block.dtype, device=moe_block.device)

        router_logits = moe_block.gate(input_batch)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, moe_block.top_k, dim=-1)
        routing_weights = routing_weights.to(moe_block.dtype)

        expert_mask = F.one_hot(selected_experts, num_classes=moe_block.number_of_experts).permute(2, 1, 0)

        experts_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_index in experts_hit:
            # TODO: захадкоженная часть.
            # !!! Работает только если все эксперты по всех блоках (только если все эксперты в нулевом блоке на самом деле) !!!
            expert_device = expert_blocks[0].device
            expert_layer = expert_blocks[0].expert_layers[int(expert_index)].to(moe_block.device)

            expert_priorities, tokens_indexes = torch.where(expert_mask[expert_index].squeeze(0))

            expert_input_batch = input_batch[tokens_indexes]
            expert_output_batch = expert_layer(expert_input_batch)
            weighted_expert_output_batch = expert_output_batch * routing_weights[tokens_indexes, expert_priorities, None]
            output_batch.index_add_(0, tokens_indexes, weighted_expert_output_batch.to(moe_block.dtype))

            expert_blocks[0].expert_layers[int(expert_index)].to(expert_device)

        output_batch = output_batch.reshape(-1, moe_block.hidden_features)

        torch.cuda.synchronize(moe_block.device)

        if not is_warmup:
            batch_size_in_bytes = batch_size * self.sequence_length * self.hidden_features * self.dtype.itemsize  # type: ignore
            self.benchmarks["e2e_moe"].stop()
            self.benchmarks["e2e_moe"].metadata["batch_size"].append(batch_size_in_bytes)
        duration = (timeit.default_timer() - duration) * 1000.0

        logger.info(f"Procced batch in on device: {output_batch.device}")
        logger.info(f"Batch size: {batch_size} tokens ({(batch_size * self.hidden_features * self.dtype.itemsize / 1024**2):.3f} MB)")  # type: ignore
        logger.info(f"Batch values: {output_batch[0][0][0]} ... {output_batch[-1][-1][-1]}")
        logger.info(f"Local MoE layer latency: ~{(duration):.5f} ms")
        print("--------------------------------------------------------------------------------------")

    def host_side(
        self,
        moe_block: DistMoEBlock,
        batch_size: int,
        input_batch: torch.Tensor,
        is_warmup: bool,
    ):
        logger.debug(
            f"Preparing for sending a batch with shape: ({batch_size // self.sequence_length}, {self.sequence_length}, {self.hidden_features}) ({batch_size} tokens; size: {batch_size * self.hidden_features * self.dtype.itemsize / 1024**2} MB)..."  # type: ignore
        )
        logger.debug(f"Batch is on {input_batch.device} device now")

        duration = timeit.default_timer()

        if not is_warmup:
            self.benchmarks["e2e_moe"].start()
        # TODO:
        old_input_shape = input_batch.shape
        result = moe_block(input_batch)
        input_batch.resize_(old_input_shape)
        duration = (timeit.default_timer() - duration) * 1000.0
        if not is_warmup:
            self.benchmarks["e2e_moe"].stop()
            batch_size_in_bytes = batch_size * self.sequence_length * self.hidden_features * self.dtype.itemsize  # type: ignore
            self.benchmarks["e2e_moe"].metadata["batch_size"].append(batch_size_in_bytes)

        logger.info(f"Procced batch in on device: {result.device}")
        logger.info(f"Procced batch shape: {result.shape}")
        logger.info(f"Batch size: {batch_size} tokens ({(batch_size * self.hidden_features * self.dtype.itemsize / 1024**2):.3f} MB)")  # type: ignore
        logger.info(f"Batch values: {result[0][0][0]} ... {result[-1][-1][-1]}")  # type: ignore
        logger.info(f"DistMoE layer latency: ~{(duration):.5f} ms")
        print("--------------------------------------------------------------------------------------")

        result.to("cpu")
        del result

    def run(self):
        self.warmup_batch_sizes.extend(self.batch_sizes)
        n = len(self.warmup_batch_sizes)

        print("--------------------------------------------------------------------------------------")
        batches = []
        for batch_size, index in zip(self.warmup_batch_sizes, range(len(self.warmup_batch_sizes))):
            batches.append(
                torch.randn((batch_size // self.sequence_length, self.sequence_length, self.hidden_features), dtype=self.dtype).to(
                    self.host_device
                )
            )
            print(f"Allocated: {index + 1}/{len(self.warmup_batch_sizes)} batches")
        print("--------------------------------------------------------------------------------------")

        for transfer_protocol in self.transfer_protocols:
            logger.info(f"Number of expert blocks: {self.number_of_expert_blocks}")

            expert_block_devices = []
            for expert_block_index in range(self.number_of_expert_blocks):
                expert_block_devices.append(torch.device(f"cuda:{self.expert_block_gpu_ids[expert_block_index]}"))

            for backend in self.backends:
                moe_backend = None
                if backend == "mcte":
                    moe_backend = TensorTransferEngineBackend.MoonCake
                elif backend == "nixl":
                    moe_backend = TensorTransferEngineBackend.NIXL

                moe_block, expert_blocks = self.setup_dist_moe(
                    moe_backend if moe_backend is not None else TensorTransferEngineBackend.NIXL,
                    self.number_of_experts,
                    self.number_of_expert_blocks,
                    expert_block_devices,
                    self.top_k,
                    transfer_protocol,
                    self.setup_ib_devices(transfer_protocol, "host"),
                    self.setup_ib_devices(transfer_protocol, "expert_block"),
                )

                for batch_index in range(n):
                    batch_size = self.warmup_batch_sizes[batch_index]

                    is_warmup = False
                    if batch_index < n - len(self.batch_sizes):
                        is_warmup = True

                    if moe_backend is None:
                        self.local_experts(moe_block, expert_blocks, batch_size, batches[batch_index], is_warmup)
                    else:
                        n = 4
                        print(f"\n\nFORWARDING {n} TIMES:")
                        for _ in range(n):
                            print("--------------------------------------------------------------------------------------")
                            self.host_side(moe_block, batch_size, batches[batch_index], is_warmup)
                        print("\n\n")

                    if not is_warmup:
                        self.collect_gpu_metrics(self.host_device.index, batch_size)

                    torch.cuda.empty_cache()
                    torch._C._cuda_clearCublasWorkspaces()

                for expert_block in expert_blocks:
                    if expert_block is not None:
                        expert_block.close()

                if moe_block is not None:
                    moe_block.close()

                for benchmark_name, benchmark in self.benchmarks.items():
                    if len(benchmark.duration) > 0:
                        benchmark.save(
                            f"/MCTE/src/python/MoE/benchmarks/data/csv/DistMoE[{self.number_of_expert_blocks}]_{transfer_protocol.value}_{backend}_{benchmark_name}"
                        )
                    benchmark.clear()

    def setup_ib_devices(self, transfer_protocol: TransferProtocol, role: str):
        ib_device_names = ""
        if transfer_protocol == TransferProtocol.RDMA:
            if role == "host":
                ib_device_names = "mlx5_2"
            elif role == "expert_block":
                ib_device_names = "mlx5_3"
        return ib_device_names

    def collect_gpu_metrics(self, gpu_id: int, batch_size: int):
        batch_size_in_bytes = batch_size * self.hidden_features * self.dtype.itemsize  # type: ignore

        total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
        reserved_memory = torch.cuda.memory_reserved(gpu_id)
        allocated_memory = torch.cuda.memory_allocated(gpu_id)

        self.benchmarks["gpu_memory"].duration.append(0)
        memory_usage = max(reserved_memory, allocated_memory) / total_memory * 100.0
        self.benchmarks["gpu_memory"].metadata["free_memory"].append(memory_usage)
        self.benchmarks["gpu_memory"].metadata["batch_size"].append(batch_size_in_bytes)

        n_gpus = torch.cuda.device_count()
        for gpu_id in range(n_gpus):
            total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
            reserved_memory = torch.cuda.memory_reserved(gpu_id)
            allocated_memory = torch.cuda.memory_allocated(gpu_id)
            memory_usage = max(reserved_memory, allocated_memory) / total_memory * 100.0
            logger.info(f"GPU {gpu_id} memory usage: ~{(memory_usage):.2f} %")


if __name__ == "__main__":
    options, batch_sizes, warmup_batch_sizes = parse_arguments()

    bench = Bench(options, batch_sizes, warmup_batch_sizes)

    logger.info(f"Start DistMoE benchmark (with different backends: nixl/mcte)...")

    duration = timeit.default_timer()
    bench.run()
    duration = timeit.default_timer() - duration

    logger.info(f"Benchmark e2e duration: ~{(duration):.3f} sec")
    logger.info(
        f"The number of benchmark batches: {len(batch_sizes)} with the number of warm-up batches: {len(warmup_batch_sizes)-len(batch_sizes)}"
    )
