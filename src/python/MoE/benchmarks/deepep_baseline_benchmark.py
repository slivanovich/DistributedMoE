#!/usr/bin/env python3
import sys
import os
import timeit
import argparse
import csv
from datetime import datetime
from pathlib import Path

sys.path.insert(0, "/MCTE/src/python")

import torch
import torch.distributed as dist
import deep_ep  # type: ignore
from MoE.MoE import Qwen3MoEMLP


hidden = 2048
num_experts = 128
top_k = 8


def setup_distributed():
    """Setup distributed environment with RDMA forced"""
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")

    # Force DeepEP to use RDMA instead of NVLink
    os.environ["USE_MNNVL"] = "0"

    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    return rank, world_size


def create_expert_models(rank, num_experts, hidden):
    """Create expert models on GPU 1 only"""
    expert_models = []
    if rank == 1:
        intermediate_size = hidden * 3
        for i in range(num_experts):
            expert = Qwen3MoEMLP(hidden, intermediate_size, torch.bfloat16)
            expert.to(f"cuda:{rank}")
            expert_models.append(expert)
        print(f"Rank {rank}: Created {len(expert_models)} expert models (ALL experts on GPU 1)")
    elif rank == 0:
        print(f"Rank {rank}: No experts on GPU 0 (data sender only)")

    return expert_models


def create_deepep_buffer(rank, world_size, batch_size, hidden, num_experts):
    """Create DeepEP buffer with proper sizing"""
    num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(batch_size, hidden, world_size, num_experts)
    # num_rdma_bytes = int(1e9)
    print(f"Rank {rank}: Calculated RDMA buffer size: {num_rdma_bytes / 1024 / 1024:.1f} MB")

    buffer = deep_ep.Buffer(
        dist.group.WORLD,
        num_rdma_bytes=num_rdma_bytes,
        low_latency_mode=True,
        num_qps_per_rank=num_experts,
        allow_nvlink_for_normal_mode=False,
        explicitly_destroy=True,
        allow_mnnvl=False,
    )

    return buffer


def create_routing_data(rank, batch_size, hidden, num_experts, top_k):
    """Create input data and routing"""
    x = torch.randn((batch_size, hidden), dtype=torch.bfloat16, device=f"cuda:{rank}")

    if rank == 0:
        scores = torch.randn((batch_size, num_experts), dtype=torch.float32, device=f"cuda:{rank}").abs() + 1
        topk_weights, topk_idx = torch.topk(scores, top_k, dim=-1, largest=True, sorted=False)
        topk_weights = torch.softmax(topk_weights, dim=-1)
        topk_idx = topk_idx.to(torch.int64)
    else:
        topk_idx = torch.empty((batch_size, top_k), dtype=torch.int64, device=f"cuda:{rank}")
        topk_weights = torch.empty((batch_size, top_k), dtype=torch.float32, device=f"cuda:{rank}")

    dist.broadcast(topk_idx, src=0)
    dist.broadcast(topk_weights, src=0)

    return x, topk_idx, topk_weights


def run_e2e_moe(buffer, x, topk_idx, topk_weights, expert_models, rank, world_size, batch_size, num_experts):
    """Run E2E MoE with expert computation"""
    # DISPATCH
    packed_recv_x, packed_recv_count, handle, event, hook = buffer.low_latency_dispatch(
        x, topk_idx, batch_size, num_experts, use_fp8=False, async_finish=True, return_recv_hook=False
    )
    event.current_stream_wait()

    # EXPERT COMPUTATION
    if rank == 0:
        simulated_gemm_x = packed_recv_x.clone()
    else:
        simulated_gemm_x = torch.zeros_like(packed_recv_x)
        experts_per_rank = num_experts // world_size
        local_expert_start = rank * experts_per_rank

        for local_expert_idx in range(experts_per_rank):
            recv_count = packed_recv_count[local_expert_idx]
            num_tokens_for_expert = recv_count.item()

            if num_tokens_for_expert > 0:
                global_expert_idx = local_expert_start + local_expert_idx
                expert_tokens = packed_recv_x[local_expert_idx, :num_tokens_for_expert]

                with torch.no_grad():
                    expert_output = expert_models[global_expert_idx](expert_tokens)

                simulated_gemm_x[local_expert_idx, :num_tokens_for_expert] = expert_output

        torch.cuda.synchronize(f"cuda:{rank}")

    dist.barrier()

    # COMBINE
    combined_x, event, hook = buffer.low_latency_combine(
        simulated_gemm_x, topk_idx, topk_weights, handle, async_finish=True, return_recv_hook=False
    )
    event.current_stream_wait()

    return combined_x


def benchmark_batch_size(rank, world_size, batch_size, hidden, num_experts, top_k, num_warmup=2, num_runs=5):
    """Benchmark a specific batch size"""
    print(f"Rank {rank}: Benchmarking batch_size={batch_size}")

    # Create expert models
    expert_models = create_expert_models(rank, num_experts, hidden)

    # Create DeepEP buffer
    buffer = create_deepep_buffer(rank, world_size, batch_size, hidden, num_experts)

    # Create routing data
    x, topk_idx, topk_weights = create_routing_data(rank, batch_size, hidden, num_experts, top_k)

    # Warmup runs
    for _ in range(num_warmup):
        _ = run_e2e_moe(buffer, x, topk_idx, topk_weights, expert_models, rank, world_size, batch_size, num_experts)

    dist.barrier()

    # Timing runs
    latencies = []
    for _ in range(num_runs):
        start_time = timeit.default_timer()
        result = run_e2e_moe(buffer, x, topk_idx, topk_weights, expert_models, rank, world_size, batch_size, num_experts)
        end_time = timeit.default_timer()

        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)
        dist.barrier()

    # Calculate statistics
    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)

    # Calculate data size and throughput
    data_size_mb = top_k * batch_size * hidden * 2 / 1024 / 1024  # bfloat16 = 2 bytes
    throughput = data_size_mb * 2 / avg_latency  # MB/ms

    # Cleanup
    try:
        buffer.destroy()
    except:
        pass

    return {
        "batch_size": batch_size,
        "avg_latency_ms": avg_latency,
        "min_latency_ms": min_latency,
        "max_latency_ms": max_latency,
        "data_size_mb": data_size_mb,
        "throughput_mb_per_ms": throughput,
        "result_shape": str(result.shape) if rank == 0 else None,  # type: ignore
    }


def save_results_to_csv(results, output_dir):
    """Save benchmark results to CSV file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"deepep_baseline_rdma_{timestamp}.csv"
    filepath = Path(output_dir) / filename

    # Ensure output directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", newline="") as csvfile:
        fieldnames = [
            "batch_size",
            "avg_latency_ms",
            "min_latency_ms",
            "max_latency_ms",
            "data_size_mb",
            "throughput_mb_per_ms",
            "result_shape",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result)

    print(f"Results saved to: {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser(description="DeepEP Baseline Benchmark")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1000, 2000, 4000, 8000, 16000], help="Batch sizes to benchmark")
    parser.add_argument("--num-warmup", type=int, default=2, help="Number of warmup runs")
    parser.add_argument("--num-runs", type=int, default=5, help="Number of timing runs")
    parser.add_argument("--output-dir", type=str, default="comprehensive_benchmark/rdma/deepep/csv", help="Output directory for CSV files")

    args = parser.parse_args()

    # Setup distributed
    rank, world_size = setup_distributed()

    if rank == 0:
        print("=== DeepEP Baseline Benchmark ===")
        print(f"Batch sizes: {args.batch_sizes}")
        print(f"Hidden: {hidden}, Experts: {num_experts}, Top-k: {top_k}")
        print(f"Warmup runs: {args.num_warmup}, Timing runs: {args.num_runs}")
        print(f"Output directory: {args.output_dir}")
        print("=" * 50)

    results = []

    for batch_size in args.batch_sizes:
        result = benchmark_batch_size(rank, world_size, batch_size, hidden, num_experts, top_k, args.num_warmup, args.num_runs)

        if rank == 0:
            results.append(result)
            print(f"Batch {batch_size}: {result['avg_latency_ms']:.2f}ms avg, " f"{result['throughput_mb_per_ms']:.1f} MB/ms throughput")

    # Save results to CSV (only rank 0)
    if rank == 0:
        csv_path = save_results_to_csv(results, args.output_dir)
        print(f"\n=== DeepEP Baseline Benchmark Complete ===")
        print(f"Results saved to: {csv_path}")

    # Cleanup
    try:
        dist.destroy_process_group()
    except:
        pass


if __name__ == "__main__":
    main()
