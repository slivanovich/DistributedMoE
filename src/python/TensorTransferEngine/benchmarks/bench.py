import time
import uuid
import torch
import signal
import timeit
import optparse
from typing import Dict, List

from TensorTransferEngine.MCTETensorTransferEngine import *
from TensorTransferEngine.NIXLTensorTransferEngine import *
from TensorTransferEngine.TensorTransferEngine import TransferProtocol
from python.TensorTransferEngine.utils import HandshakeType
from utils.utils import get_p2p_ports
from utils.Benchmark import Benchmark
from utils.logging_config import get_logger

logger = get_logger("dist_moe.benchmarks.tte")


def parse_arguments():
    parser = optparse.OptionParser()
    parser.add_option(
        "--mode",
        dest="mode",
        help="specify a mode of the TT instance (target/initiator)",
        nargs=1,
        type="string",
    )
    parser.add_option(
        "--local_address",
        dest="local_address",
        help="specify the local server full address (with port)",
        nargs=1,
        type="str",
    )
    parser.add_option(
        "--remote_address",
        dest="remote_address",
        help="specify the remote server full address (with port)",
        nargs=1,
        type="str",
    )
    parser.add_option(
        "--ib_device_names",
        dest="ib_device_names",
        help="specify the IB devices names (mlx5_0/mlx5_0,mlx5_1,mlx5_2/[empty string])",
        nargs=1,
        type="str",
        default="",
    )
    parser.add_option(
        "--warmup_tensor_size",
        dest="warmup_tensor_size",
        help="specify the sizes of a warm up tensors",
        nargs=1,
        type="str",
        default="65536,65536,65536,65536",  # in bytes
    )
    parser.add_option(
        "--tensor_size",
        dest="tensor_size",
        help="specify the sizes of a tensors",
        nargs=1,
        type="str",
        default="256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,524288,1048576,2097152,4194304,8388608,16777216,33554432,67108864,104857600,209715200,314572800,419430400,524288000,1048576000,1153433600,1258291200,1363148800,1468006400,1572864000",  # in bytes
    )
    parser.add_option(
        "--gpu_id",
        dest="gpu_id",
        help="specify the gpu's id (from 0 to 7)",
        nargs=1,
        type="int",
        default=0,
    )
    parser.add_option(
        "--transfer_protocol",
        dest="transfer_protocol",
        help="specify the transfer protocol (rdma/nvlink)",
        nargs=1,
        type="string",
    )

    (options, _) = parser.parse_args()
    tensor_sizes = list(map(int, options.tensor_size.split(",")))
    warmup_tensor_sizes = list(map(int, options.warmup_tensor_size.split(",")))

    return (options, tensor_sizes, warmup_tensor_sizes)


class Bench:
    def __init__(self, options, tensor_sizes: list[int], warmup_tensor_sizes: list[int]) -> None:
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            logger.info(f"Number of available CUDA devices: {n_gpus}")
            for i in range(n_gpus):
                logger.info(f"Device {i}: {torch.cuda.get_device_name(i)}")
            self.device = torch.device(f"cuda:{options.gpu_id}")
        else:
            logger.warning("CUDA is not available")
            self.device = torch.device("cpu")

        self.benchmarks: Dict[str, Benchmark] = {}
        self.benchmarks["e2e_get"] = Benchmark()
        self.benchmarks["e2e_get"].metadata["data_size"] = []
        self.benchmarks["e2e_send"] = Benchmark()
        self.benchmarks["e2e_send"].metadata["data_size"] = []
        self.benchmarks["register"] = Benchmark()
        self.benchmarks["register"].metadata["data_size"] = []
        self.benchmarks["deregister"] = Benchmark()
        self.benchmarks["deregister"].metadata["data_size"] = []
        self.benchmarks["throughput"] = Benchmark()
        self.benchmarks["throughput"].metadata["data_size"] = []
        self.benchmarks["throughput"].metadata["throughput"] = []

        self.mode = options.mode
        self.warmup_tensor_sizes = warmup_tensor_sizes
        self.tensor_sizes = tensor_sizes

        src_host, src_port = options.local_address.split(":")
        dst_host, dst_port = options.remote_address.split(":")

        self.mcte_config = {
            "src_host": src_host,
            "src_port": int(src_port),
            "src_p2p_ports": get_p2p_ports(int(src_port), [int(dst_port)]),
            "dst_hosts": [dst_host],
            "dst_ports": [int(dst_port)],
            "dst_p2p_ports": get_p2p_ports(int(dst_port), [int(dst_port)]),
            "metadata_scheme": "http",
            "metadata_host": "localhost",
            "metadata_port": 8080,
            "metadata_dir": "metadata",
            "handshake_type": HandshakeType.NONE,
            "transfer_protocol": TransferProtocol.RDMA if options.transfer_protocol == "rdma" else TransferProtocol.NVLINK,
            "ib_device_names": options.ib_device_names,
        }
        self.nixl_config = {
            "src_host": src_host,
            "src_port": int(src_port),
            "src_p2p_ports": get_p2p_ports(int(src_port), [int(dst_port)]),
            "dst_hosts": [dst_host],
            "dst_ports": [int(dst_port)],
            "dst_p2p_ports": get_p2p_ports(int(dst_port), [int(dst_port)]),
            "handshake_type": HandshakeType.NONE,
            "transfer_protocol": TransferProtocol.RDMA if options.transfer_protocol == "rdma" else TransferProtocol.NVLINK,
            "ib_device_names": options.ib_device_names,
        }

        self.transfer_ways = [
            "nixl_reading",
            "nixl_writing",
            "mcte_reading",
            "mcte_writing",
        ]

        self.executor = None

        # ))))))))))))))))))))))))))))))))))))))))))))))))
        # signal.signal(signal.SIGINT, self.close)
        # signal.signal(signal.SIGTERM, self.close)
        # signal.signal(signal.SIGQUIT, self.close)

    def wait_for_transfer(self, transfer_name: bytes):
        assert self.executor is not None
        success = self.executor.wait_for_transfer_efficient(transfer_name, 3000)
        if not success:
            logger.error(f"Transfer {transfer_name.hex()[:8]} timed out after 60s")
            raise TimeoutError(f"Transfer timed out")

    def initiator_side(self, transfer_name: bytes, transfer_way: str, tensor_size: int, is_warmup: bool):
        assert self.executor is not None

        logger.debug(f"Preparing for sending tensor with size: {tensor_size / 1024**2} MB...")

        tensor = torch.randn(tensor_size // 4, dtype=torch.float32).to(self.device)
        logger.debug(f"Tensor is on {tensor.device} device now")

        if not is_warmup:
            self.benchmarks["register"].start()
        src_data_transfer_descs, src_data_registration_descs = self.executor.register_batch(
            [tensor], "VRAM" if self.device.type == "cuda" else "DRAM", self.device.index
        )
        if not is_warmup:
            self.benchmarks["register"].stop()
        if transfer_way.split("_")[1] == "writing":
            header, meta = self.executor.get_metadata_from(self.executor.remote_names[0])
            assert header == 38
            dst_data_transfer_descs = self.executor.deserialize_descs(meta[0])
            logger.debug(f"Start a transfer with name: {transfer_name}")
            self.executor.write_batch_transfer(
                self.executor.remote_names[0],
                transfer_name,
                src_data_transfer_descs,
                dst_data_transfer_descs,
                [tensor_size],
            )
            self.wait_for_transfer(transfer_name)
        else:
            dst_data_serialized_transfer_descs = self.executor.serialize_descs(src_data_transfer_descs)
            self.executor.send_metadata(38, [dst_data_serialized_transfer_descs])  # type: ignore
            header, meta = self.executor.get_metadata_from(self.executor.remote_names[0])
            assert header == 4738
        if not is_warmup:
            self.benchmarks["deregister"].start()
        self.executor.deregister_batch(src_data_registration_descs)
        if not is_warmup:
            self.benchmarks["deregister"].stop()
            self.benchmarks["register"].metadata["data_size"].append(tensor_size)
            self.benchmarks["deregister"].metadata["data_size"].append(tensor_size)

    def target_side(self, transfer_name: bytes, transfer_way: str, tensor_size: int, is_warmup: bool):
        assert self.executor is not None

        logger.info("Target host listening...")

        tensor = torch.zeros(tensor_size // 4, dtype=torch.float32).to(self.device)

        dst_data_transfer_descs, dst_data_registration_descs = self.executor.register_batch(
            [tensor], "VRAM" if self.device.type == "cuda" else "DRAM", self.device.index
        )
        if transfer_way.split("_")[1] == "reading":
            header, meta = self.executor.get_metadata_from(self.executor.remote_names[0])
            assert header == 38
            src_data_transfer_descs = self.executor.deserialize_descs(meta[0])
            logger.debug(f"Start a transfer with name: {transfer_name}")
            self.executor.read_batch_transfer(
                self.executor.remote_names[0],
                transfer_name,
                src_data_transfer_descs,
                dst_data_transfer_descs,
                [tensor_size],
            )
            self.wait_for_transfer(transfer_name)
        else:
            dst_data_serialized_transfer_descs = self.executor.serialize_descs(dst_data_transfer_descs)
            self.executor.send_metadata(38, [dst_data_serialized_transfer_descs])  # type: ignore
            header, meta = self.executor.get_metadata_from(self.executor.remote_names[0])
            assert header == 4738
        self.executor.deregister_batch(dst_data_registration_descs)

        logger.debug(
            f"Have got the tensor: {tensor[0]}, ..., {tensor[-1]}\n \
            Tensor is on {tensor.device} device now\n \
            Tensor size: {tensor_size / (1024**2)} MB"
        )

    def run(self):
        self.warmup_tensor_sizes.extend(self.tensor_sizes)
        n = len(self.warmup_tensor_sizes)

        for transfer_way in self.transfer_ways:
            if transfer_way[:4] == "nixl":
                self.executor = NIXLTensorTransferEngine(**self.nixl_config)
            elif transfer_way[:4] == "mcte":
                self.executor = MCTETensorTransferEngine(**self.mcte_config)

            if self.executor is None:
                raise RuntimeError(
                    f"There is no such an implementation of the TensorTransfer for the {options.transfer_way} transfer way, aborting..."
                )

            print("--------------------------------------------------------------------------------------")
            duration = timeit.default_timer()
            logger.info(f"The {transfer_way} tensor transfer executor has been opened...")
            logger.info(f"Working on {self.device} device.")
            print("--------------------------------------------------------------------------------------")

            warmup_transfers: List[bytes] = []
            for index in range(n):
                transfer_name = uuid.uuid4().bytes
                tensor_size = self.warmup_tensor_sizes[index]

                is_warmup = False
                if index < n - len(self.tensor_sizes):
                    is_warmup = True
                    warmup_transfers.append(transfer_name)

                if self.mode == "initiator":
                    self.initiator_side(transfer_name, transfer_way, tensor_size, is_warmup)
                elif self.mode == "target":
                    self.target_side(transfer_name, transfer_way, tensor_size, is_warmup)

                with self.executor._transfers_lock:
                    for done_transfer in self.executor.done_transfers.values():
                        if not is_warmup and done_transfer.name == transfer_name:
                            self.benchmarks["throughput"].duration.append(done_transfer.duration)
                            self.benchmarks["throughput"].metadata["data_size"].append(tensor_size)
                            self.benchmarks["throughput"].metadata["throughput"].append(tensor_size / 1024**3 / done_transfer.duration)

            for benchmark_name, benchmark in self.benchmarks.items():
                if len(benchmark.duration) > 0:
                    benchmark.save(f"/MCTE/src/python/TensorTransfer/benchmarks/data/csv/TT_{transfer_way}_{benchmark_name}")
                benchmark.clear()

            self.executor.close()
            self.executor = None

            print("--------------------------------------------------------------------------------------")
            duration = timeit.default_timer() - duration
            logger.info(f"Duration: ~{(duration)} sec")
            logger.info(f"The {transfer_way} executor has been closed...")
            print("--------------------------------------------------------------------------------------")

    def close(self, signum, frame):
        if self.executor is not None:
            self.executor.close()


if __name__ == "__main__":
    options, tensor_sizes, warmup_tensor_sizes = parse_arguments()

    # bench = Bench(options, tensor_sizes, warmup_tensor_sizes)

    logger.info(f"LEGACY, aborting...")
    # logger.info(f"Start TensorTranfer benchmark (with differnent transfer ways)...")

    # duration = timeit.default_timer()
    # bench.run()
    # duration = timeit.default_timer() - duration

    # logger.info(f"Benchmark e2e duration: ~{(duration):.3f} sec")
    # logger.info(
    #     f"The number of benchmark buffers: {len(tensor_sizes)} with the number of warm-up buffers: {len(warmup_tensor_sizes)-len(tensor_sizes)}"
    # )

    # bench.close(0, 0)
