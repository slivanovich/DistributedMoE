import copy
from threading import Thread
from torch import nn
from torch.profiler import ProfilerActivity
from typing import List, Tuple
import os
import pandas as pd
import time

import numpy as np
import pytest
import sys
import timeit
import torch
import torch.nn.functional as F

from TensorTransferEngine.TensorTransferEngine import TensorTransferEngineBackend, TensorTransferEngineConfig
from TensorTransferEngine.utils import HandshakeType, TransferProtocol

from MoE.MoE import Qwen3MoEMLP, Qwen3SparseMoEBlock
from MoE.DistExpertsBlock import DistExpertsBlock
from MoE.DistMoEBlock import DistMoEBlock, DistMoEBlockConfig
from MoE.MetricsCollector import MetricsCollector, MetricsCollectorConfig

host_port = 10000
expert_block_ports_start_index = 50000

number_of_expert_blocks = 1
number_of_experts = 128
top_k = 8

sequence_length = 1524

io_buffer_size = top_k * 1024
number_of_io_buffers = 32
hidden_features = 2048
intermediate_features = 6144

using_alignment = False


def nottest(obj):
    obj.__test__ = False
    return obj


@pytest.fixture
def Qwen3SparseMoeBlock_instance(host_device, dtype):
    moe_block = Qwen3SparseMoEBlock(number_of_experts, top_k, hidden_features, intermediate_features, dtype=dtype)
    moe_block.to(host_device)

    return moe_block


@pytest.fixture
def DistMoEBlock_instance(
    host_device, remote_device, backend, dtype
) -> Tuple[DistMoEBlock, List[Qwen3MoEMLP], MetricsCollector, List[Thread]]:
    assert host_port < expert_block_ports_start_index

    host_p2p_ports = [host_port + 1000 * (index + 1) for index in range(number_of_expert_blocks)]

    expert_block_hosts: List[str] = []
    expert_block_ports: List[int] = []
    expert_block_p2p_ports: List[int] = []
    expert_blocks_experts: List[List[int]] = []
    expert_block_devices: List[torch.device] = []

    for experts_block_index in range(number_of_expert_blocks):
        expert_block_hosts.append("localhost")
        expert_block_ports.append(expert_block_ports_start_index + 100 * experts_block_index)
        expert_block_p2p_ports.append(expert_block_ports_start_index + 100 * experts_block_index + 1000)
        expert_block_devices.append(remote_device)
        expert_blocks_experts.append([])

    # Distribute experts evenly across expert blocks.
    for expert_index in range(number_of_experts):
        block_index = expert_index % number_of_expert_blocks
        expert_blocks_experts[block_index].append(expert_index)

    # Initialize metrics collector.
    metrics_config = MetricsCollectorConfig(
        enable_collection=True,
        enable_detailed_timing=True,
        enable_transfer_metrics=True,
        enable_expert_block_metrics=True,
        max_stored_forwards=10000,
        export_directory="./test_metrics_output",
    )
    metrics_collector = MetricsCollector(metrics_config)

    # Health check ports allocation (avoiding collisions with existing ports).
    host_health_check_ports = [host_port + 2000 * (index + 1) for index in range(number_of_expert_blocks)]
    remote_health_check_ports = [expert_block_ports_start_index + 100 * index + 2000 for index in range(number_of_expert_blocks)]

    dist_moe_config = DistMoEBlockConfig(
        backend=backend,
        expert_blocks=expert_blocks_experts,
        number_of_experts=number_of_experts,
        top_k=top_k,
        number_of_io_buffers=number_of_io_buffers,
        io_buffer_size=io_buffer_size,
        hidden_features=hidden_features,
        dtype=dtype,
        moe_device=host_device,
        eb_devices=expert_block_devices,
        forward_timeout=4000,
        host_health_check_ports=host_health_check_ports,
        remote_health_check_ports=remote_health_check_ports,
    )

    expert_layers = [Qwen3MoEMLP(hidden_features, intermediate_features, dtype) for _ in range(number_of_experts)]
    expert_block_threads = []

    for experts_block_index in range(number_of_expert_blocks):
        experts_block_layers = []
        for expert_index in expert_blocks_experts[experts_block_index]:
            experts_block_layers.append(copy.deepcopy(expert_layers[expert_index]))

        eb_tte_config = TensorTransferEngineConfig(
            host_address=expert_block_hosts[experts_block_index],
            host_port=expert_block_ports[experts_block_index],
            host_p2p_ports=[expert_block_p2p_ports[experts_block_index]],
            remote_addresses=["localhost"],
            remote_ports=[host_port],
            remote_p2p_ports=[host_p2p_ports[experts_block_index]],
            handshake_type=HandshakeType.CLIENT,
            transfer_protocol=TransferProtocol.RDMA,
            ib_device_names="mlx5_5",
            p2p_timeout_duration=3000,
            metadata_schema="http",
            metadata_host="localhost",
            metadata_port=8080,
            metadata_dir="metadata",
        )

        def thread_job():
            dist_experts_block = DistExpertsBlock(
                experts_block_index, dist_moe_config, experts_block_layers, eb_tte_config, metrics_collector
            )
            dist_experts_block.main_loop()
            dist_experts_block.close()

        t = Thread(target=thread_job, daemon=False)  # Changed to non-daemon
        t.start()
        expert_block_threads.append(t)

    dist_moe_tte_config = TensorTransferEngineConfig(
        host_address="localhost",
        host_port=host_port,
        host_p2p_ports=host_p2p_ports,
        remote_addresses=expert_block_hosts,
        remote_ports=expert_block_ports,
        remote_p2p_ports=expert_block_p2p_ports,
        handshake_type=HandshakeType.SERVER,
        transfer_protocol=TransferProtocol.RDMA,
        ib_device_names="mlx5_4",
        p2p_timeout_duration=3000,
        metadata_schema="http",
        metadata_host="localhost",
        metadata_port=8080,
        metadata_dir="metadata",
    )

    # Create DistMoEBlock with metrics collector
    moe_block = DistMoEBlock(dist_moe_config, dist_moe_tte_config, metrics_collector)
    moe_block.to(host_device)

    return moe_block, expert_layers, metrics_collector, expert_block_threads


@nottest
def local_experts(moe_block: DistMoEBlock, expert_layers: List[Qwen3MoEMLP], batch: torch.Tensor):
    shape = batch.shape
    input_batch = batch.view(-1, moe_block.hidden_features)
    output_batch = torch.zeros(input_batch.shape, dtype=moe_block.dtype, device=moe_block.device)

    router_logits = moe_block.gate(input_batch)
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, moe_block.top_k, dim=-1)
    routing_weights = routing_weights.to(moe_block.dtype)

    expert_mask = F.one_hot(selected_experts, num_classes=moe_block.number_of_experts).permute(2, 1, 0)

    experts_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_index in experts_hit:
        expert_layer = expert_layers[int(expert_index)].to(moe_block.device)

        expert_priorities, tokens_indexes = torch.where(expert_mask[expert_index].squeeze(0))

        expert_input_batch = input_batch[tokens_indexes]
        expert_output_batch = expert_layer(expert_input_batch)
        weighted_expert_output_batch = expert_output_batch * routing_weights[tokens_indexes, expert_priorities, None]
        output_batch.index_add_(0, tokens_indexes, weighted_expert_output_batch.to(moe_block.dtype))

    output_batch = output_batch.reshape(shape)
    return output_batch


def run_bench_with(
    batch_size: int,
    sequence_length: int,
    host_device: torch.device,
    dtype: torch.dtype,
    moe_block: DistMoEBlock,
):
    num_warmup_runs = 32
    num_perf_runs = 32

    batch_shape = (batch_size // sequence_length, sequence_length, hidden_features)
    input_batch = torch.randn(size=batch_shape, dtype=dtype, device=host_device)

    for _ in range(num_warmup_runs):
        after_dist_moe: torch.Tensor = moe_block(input_batch)

    # Clear metrics after warmup to only measure performance runs.
    if moe_block.metrics_collector is not None:
        moe_block.metrics_collector.clear_metrics()

    dist_moe_durations = []
    for _ in range(num_perf_runs):
        dist_moe_duration = timeit.default_timer()
        after_dist_moe = moe_block(input_batch, using_alignment=using_alignment)
        dist_moe_duration = (timeit.default_timer() - dist_moe_duration) * 1000.0
        dist_moe_durations.append(dist_moe_duration)

    def calculate_stats(durations):
        durations_np = np.array(durations)
        return {
            "mean": np.mean(durations_np),
            "p01": np.percentile(durations_np, 1),
            "p50": np.percentile(durations_np, 50),
            "p95": np.percentile(durations_np, 95),
            "p99": np.percentile(durations_np, 99),
        }

    dist_stats = calculate_stats(dist_moe_durations)

    # Total memory amount that have been transferred (and computed).
    transferred_memory_mb = top_k * batch_size * hidden_features * dtype.itemsize / 1024**2

    # Calculate theoretical baseline
    ib_bandwith = 50  # in MB/ms
    total_compute_duration = 1  # in ms
    theoretical_estimation = (top_k * batch_size * hidden_features * dtype.itemsize / 1024**2) / ib_bandwith * 2 + total_compute_duration
    speedup = 100 - 100 * dist_stats["mean"] / theoretical_estimation

    # Prepare data for CSV.
    stats_data = {
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "transferred_memory_mb": transferred_memory_mb,
        "mean_latency_ms": dist_stats["mean"],
        "p01_latency_ms": dist_stats["p01"],
        "p50_latency_ms": dist_stats["p50"],
        "p95_latency_ms": dist_stats["p95"],
        "p99_latency_ms": dist_stats["p99"],
        "theoretical_latency_ms": theoretical_estimation,
        "ib_bandwidth_mb_ms": ib_bandwith,
        "compute_duration_ms": total_compute_duration,
        "speedup_percent": speedup,
        "num_expert_blocks": number_of_expert_blocks,
        "top_k": top_k,
        "hidden_features": hidden_features,
        "intermediate_features": intermediate_features,
        "warmup_runs": num_warmup_runs,
        "perf_runs": num_perf_runs,
        "device": str(host_device),
        "dtype": str(dtype),
    }

    # Statistics collection and saving.
    data_dir = "/MCTE/src/python/MoE/tests/data"
    csv_dir = f"{data_dir}/csv"
    os.makedirs(csv_dir, exist_ok=True)

    # Save individual run statistics to CSV.
    csv_filename = f"{csv_dir}/moe_benchmark_batch_{batch_size}_seq_{sequence_length}.csv"
    df = pd.DataFrame([stats_data])
    df.to_csv(csv_filename, index=False)

    # Also append to a combined statistics file.
    combined_csv = f"{csv_dir}/moe_benchmark_combined.csv"
    if os.path.exists(combined_csv):
        existing_df = pd.read_csv(combined_csv)
        # Remove any existing entries with the same batch_size to avoid duplicates
        existing_df = existing_df[existing_df["batch_size"] != batch_size]
        combined_df = pd.concat([existing_df, df], ignore_index=True)
    else:
        combined_df = df
    # Sort by batch_size for consistent ordering
    combined_df = combined_df.sort_values("batch_size").reset_index(drop=True)
    combined_df.to_csv(combined_csv, index=False)

    print(f"Statistics saved to: {csv_filename}")
    print(f"Combined statistics updated: {combined_csv}")

    width = len("------------------MoE durations------------------")
    separator_line = "-" * width

    print(separator_line)
    print(f"{'MoE Performance Benchmark':^{width}}")
    print(separator_line)
    print(f"{'EB:':<35} {number_of_expert_blocks} blocks")
    print(f"{'TopK:':<35} {top_k} experts")
    print(f"{'Batch size:':<35} {batch_size} tokens")
    print(f"{'Transferred memory:':<35} {transferred_memory_mb:.3f} MB")
    print(f"{'Warmup runs:':<35} {num_warmup_runs}")
    print(f"{'Performance runs:':<35} {num_perf_runs}")
    print(separator_line)

    print(f"{'Distributed MoE layer E2E Latency':^{width}}")
    print(separator_line)
    print(f"{'Avg:':<35} {dist_stats['mean']:.2f} ms")
    print(f"{'P01:':<35} {dist_stats['p01']:.2f} ms")
    print(f"{'P50:':<35} {dist_stats['p50']:.2f} ms")
    print(f"{'P95:':<35} {dist_stats['p95']:.2f} ms")
    print(f"{'P99:':<35} {dist_stats['p99']:.2f} ms")
    print(separator_line)

    # Display and export metrics if metrics collector is available.
    if moe_block.metrics_collector is not None:
        print(f"{'Metrics Analysis':^{width}}")
        print(separator_line)

        try:
            # Get the raw metrics that match Metrics.py classes structure
            metrics = moe_block.metrics_collector.get_metrics()

            if metrics and metrics.number_of_forwards > 0:
                # Aggregate statistics across all forwards
                total_e2e_latency = sum(f.e2e_latency for f in metrics.forwards)
                total_setup_latency = sum(f.setup_latency for f in metrics.forwards)
                total_avg_dispatches_latency = sum(f.avg_dispatches_e2e_latency for f in metrics.forwards)
                total_avg_combines_latency = sum(f.avg_combines_e2e_latency for f in metrics.forwards)

                # Count total operations
                total_dispatches = sum(len(f.dispatches) for f in metrics.forwards)
                total_combines = sum(len(f.combines) for f in metrics.forwards)
                total_expert_runs = 0

                # Aggregate expert run statistics
                all_expert_run_latencies = []
                all_read_throughputs = []
                all_write_throughputs = []

                for forward in metrics.forwards:
                    for eb in forward.experts_blocks:
                        for expert_run in eb.experts_runs:
                            total_expert_runs += 1
                            all_expert_run_latencies.append(expert_run.e2e_latency)
                            all_read_throughputs.append(expert_run.read_transfer_throughput)
                            all_write_throughputs.append(expert_run.write_transfer_throughput)

                avg_expert_run_latency = sum(all_expert_run_latencies) / len(all_expert_run_latencies)

                # Display aggregated forward metrics
                print(f"{'Aggregated Forward Metrics':^{width}}")
                print(separator_line)
                print(f"{'Avg E2E Latency:':<35} {total_e2e_latency / metrics.number_of_forwards:.2f} ms")
                print(f"{'Avg Setup Latency:':<35} {total_setup_latency / metrics.number_of_forwards:.2f} ms")
                print(f"{'Avg Dispatches E2E Latency:':<35} {total_avg_dispatches_latency / metrics.number_of_forwards:.2f} ms")
                print(f"{'Avg Combines E2E Latency:':<35} {total_avg_combines_latency / metrics.number_of_forwards:.2f} ms")
                print(f"{'Avg Experts Run Latency:':<35} {avg_expert_run_latency:.2f} ms")
                print(separator_line)

                # Display operation counts
                print(f"{'Operation Counts':^{width}}")
                print(separator_line)
                print(f"{'Total Dispatches:':<35} {total_dispatches}")
                print(f"{'Total Combines:':<35} {total_combines}")
                print(f"{'Total Experts Runs:':<35} {total_expert_runs}")
                print(f"{'Avg Dispatches per Forward:':<35} {total_dispatches / metrics.number_of_forwards:.1f}")
                print(f"{'Avg Combines per Forward:':<35} {total_combines / metrics.number_of_forwards:.1f}")
                print(f"{'Avg Experts Runs per Forward:':<35} {total_expert_runs / metrics.number_of_forwards:.1f}")
                print(separator_line)

                # Display aggregated expert run statistics
                avg_read_throughput = 0
                avg_write_throughput = 0
                if all_expert_run_latencies:
                    avg_read_throughput = sum(all_read_throughputs) / len(all_read_throughputs)
                    avg_write_throughput = sum(all_write_throughputs) / len(all_write_throughputs)

                print(f"{'Data Transfers Statistics':^{width}}")
                print(separator_line)
                print(f"{'Avg Read Throughput:':<35} {avg_read_throughput:.2f} MB/ms")
                print(f"{'Avg Write Throughput:':<35} {avg_write_throughput:.2f} MB/ms")
                avg_thoughput = (avg_read_throughput + avg_write_throughput) / 2
                practical_ib_bandwith = avg_thoughput * ((total_dispatches / metrics.number_of_forwards + 1) / 2)
                print(f"{'Practical IB Bandwidth:':<35} {practical_ib_bandwith:.2f} MB/ms")
                print(separator_line)

                print(f"{'Performance Comparison':^{width}}")
                print(separator_line)
                ib_bandwith = 50  # in MB/ms
                transfers_time_theoretical = (top_k * batch_size * hidden_features * dtype.itemsize / 1024**2) / ib_bandwith * 2
                total_compute_duration = 1  # in ms
                theoretical_estimation = transfers_time_theoretical + total_compute_duration
                speedup = 100 - 100 * dist_stats["mean"] / theoretical_estimation

                print(f"{'Pure Theoretical':^{width}}")
                print(f"{'IB Bandwith:':<35} {ib_bandwith:.2f} MB/ms")
                print(f"{'Total Transfers Time:':<35} {transfers_time_theoretical:.2f} ms")
                print(f"{'Compute Duration:':<35} {total_compute_duration} ms")
                print(f"{'Estimation:':<35} {theoretical_estimation:.2f} ms")
                print(f"{'Avg Speedup:':<35} {speedup:.2f}%")
                print(separator_line)

                if practical_ib_bandwith > 0:
                    transfers_time_theoretical = (
                        (top_k * batch_size * hidden_features * dtype.itemsize / 1024**2) / practical_ib_bandwith * 2
                    )
                    measured_theoretical_estimation = transfers_time_theoretical + total_compute_duration
                    measured_speedup = 100 - 100 * dist_stats["mean"] / measured_theoretical_estimation

                    print(f"{'Measured Theoretical':^{width}}")
                    print(f"{'IB Bandwith:':<35} {practical_ib_bandwith:.2f} MB/ms")
                    print(f"{'Total Transfers Time:':<35} {transfers_time_theoretical:.2f} ms")
                    print(f"{'Compute Duration:':<35} {total_compute_duration} ms")
                    print(f"{'Estimation:':<35} {measured_theoretical_estimation:.2f} ms")
                    print(f"{'Avg Speedup:':<35} {measured_speedup:.2f}%")
                    print(separator_line)

                    print(f"{'Overhead Analysis in Experts Blocks':^{width}}")
                    print(separator_line)
                    avg_transfers_time_in_run = (moe_block.io_buffer_size * hidden_features * dtype.itemsize / 1024**2) / avg_thoughput * 2
                    overhead = avg_expert_run_latency - avg_transfers_time_in_run
                    print(f"{'Avg Experts Run Latency:':<35} {avg_expert_run_latency:.2f} ms")
                    print(f"{'Avg Read&Write Throughput:':<35} {avg_thoughput:.2f} MB/ms")
                    print(f"{'Avg Read&Write Time:':<35} {avg_transfers_time_in_run:.2f} ms")
                    print(f"{'Overhead:':<35} {overhead:.2f} ms")
                    print(separator_line)

                # Export JSON metrics for this batch size.
                try:
                    json_file = moe_block.metrics_collector.export_metrics_json_only(f"batch_{batch_size}_seq_{sequence_length}")
                    if json_file:
                        print(f"{'Metrics exported to JSON:':<35} {json_file}")
                    else:
                        print(f"{'JSON export failed:':<35} No JSON file created")
                except Exception as e:
                    print(f"{'Metrics export failed:':<35} {e}")
            else:
                print(f"{'No metrics collected':<35}")

        except Exception as e:
            print(f"{'Metrics analysis failed:':<35} {e}")

        print(separator_line)


def test_forward(DistMoEBlock_instance, host_device, dtype):
    moe_block, expert_layers, metrics_collector, expert_block_threads = DistMoEBlock_instance

    # activities = [ProfilerActivity.CPU]
    # if torch.cuda.is_available():
    #     activities += [ProfilerActivity.CUDA]
    # else:
    #     print("Neither CUDA nor XPU devices are available to demonstrate profiling on acceleration devices")
    #     exit(0)

    # with profile(activities=activities, record_shapes=True) as prof:
    #     with record_function("model_inference"):

    # with profile(activities=activities, record_shapes=True) as prof:
    #     with record_function("model_inference"):
    #         after_dist_moe = moe_block(input_batch)

    print(f"Running benchmarck with dfferent context lengths and correctness test.\n\n")
    print(f"I/O buffers size: {io_buffer_size}")
    print(f"I/O buffers number: {number_of_io_buffers}")
    print(f"Hidden features: {hidden_features}")
    print(f"Intermediate features: {intermediate_features}")
    print("\n\n")

    for batch_size in range(sequence_length, 16 * sequence_length + 1, sequence_length):
        run_bench_with(batch_size, sequence_length, host_device, dtype, moe_block)

    # Correctness test.
    batch_size = 16000
    batch_shape = (batch_size // sequence_length, sequence_length, hidden_features)
    input_batch = torch.randn(size=batch_shape, dtype=dtype, device=host_device) + 10

    after_dist_moe = moe_block(input_batch)
    after_local_moe = local_experts(moe_block, expert_layers, input_batch)

    print(after_local_moe.view(-1, hidden_features))
    print(after_dist_moe.view(-1, hidden_features))

    try:
        assert isinstance(after_local_moe, torch.Tensor)
        assert isinstance(after_dist_moe, torch.Tensor)
        assert after_local_moe.device == host_device
        assert after_dist_moe.device == host_device
        assert after_local_moe.shape == after_dist_moe.shape
        print(f"ALL CLOSE? {torch.allclose(after_dist_moe, after_local_moe, atol=1e-1, rtol=1e-2)}")
        assert torch.allclose(after_dist_moe, after_local_moe, atol=1e-1, rtol=1e-2)
    finally:
        moe_block.close()

        # Wait for expert block threads to finish
        for thread in expert_block_threads:
            thread.join(timeout=10.0)  # Wait up to 10 seconds for each thread


@pytest.fixture(scope="session")
def backend(request):
    backend_option = request.config.getoption("--backend")
    if backend_option == "nixl":
        return TensorTransferEngineBackend.NIXL
    elif backend_option == "mcte":
        return TensorTransferEngineBackend.MoonCake


@pytest.fixture(scope="session")
def host_device(request):
    device_id = request.config.getoption("--gpu_id_host")
    if device_id == "cpu":
        return torch.device(f"cpu")
    device_id = int(device_id)
    return torch.device(f"cuda:{device_id}")


@pytest.fixture(scope="session")
def remote_device(request):
    device_id = request.config.getoption("--gpu_id_remote")
    if device_id == "cpu":
        return torch.device(f"cpu")
    device_id = int(device_id)
    return torch.device(f"cuda:{device_id}")


@pytest.fixture(scope="session")
def dtype(request):
    precision = request.config.getoption("--precision")
    if precision == "fp16":
        return torch.float16
    elif precision == "fp32":
        return torch.float32
    elif precision == "int8":
        return torch.int8
    elif precision == "bf16":
        return torch.bfloat16


# @pytest.fixture(scope="session")
# def schema(request):
#     return request.config.getoption("--schema")


if __name__ == "__main__":
    pytest.main(sys.argv)
