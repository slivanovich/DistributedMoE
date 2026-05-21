#!/usr/bin/env python3
import sys
import os
import timeit

sys.path.insert(0, "/MCTE/src/python")

import torch
import torch.distributed as dist
import deep_ep  # type: ignore
from MoE.MoE import Qwen3MoEMLP  # type: ignore

# Setup distributed with FORCED RDMA communication
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "12355")

# Environment variables to force RDMA/InfiniBand usage
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# os.environ["NVSHMEM_HCA_LIST"] = "mlx5_0:1"
# os.environ["NVSHMEM_DISABLE_NVLS"] = "1"
# os.environ["NCCL_IB_HCA"] = "mlx5_0:1"
# os.environ["NCCL_NET_GDR_LEVEL"] = "2"
# os.environ["NCCL_IB_DISABLE"] = "0"
# os.environ["NCCL_NVB_DISABLE"] = "1"

dist.init_process_group(backend="nccl")

# Force DeepEP to use RDMA instead of NVLink
os.environ["USE_MNNVL"] = "0"

rank = dist.get_rank()
world_size = dist.get_world_size()
torch.cuda.set_device(rank)

print(f"Rank {rank}: DeepEP RDMA E2E MoE test starting...")

# MoE configuration
batch_size = 16000
hidden = 2048
num_experts = 128
top_k = 8

# Create ALL expert models on GPU 1 only (GPU 0 has no experts)
expert_models = []
if rank == 1:
    intermediate_size = hidden * 3
    for i in range(num_experts):  # Create ALL experts on GPU 1
        expert = Qwen3MoEMLP(hidden, intermediate_size, torch.bfloat16)
        expert.to(f"cuda:{rank}")
        expert_models.append(expert)
    print(f"Rank {rank}: Created {len(expert_models)} expert models (ALL experts on GPU 1)")
elif rank == 0:
    print(f"Rank {rank}: No experts on GPU 0 (data sender only)")

# Calculate proper buffer sizes using DeepEP's size hint
num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(batch_size, hidden, world_size, num_experts)
print(f"Rank {rank}: Calculated RDMA buffer size: {num_rdma_bytes / 1024 / 1024:.1f} MB")

# Create DeepEP buffer with RDMA forced (key insight from test_normal_without_nvl.py)
buffer = deep_ep.Buffer(
    dist.group.WORLD,
    num_rdma_bytes=num_rdma_bytes,
    low_latency_mode=True,
    num_qps_per_rank=num_experts,
    allow_nvlink_for_normal_mode=False,  # Force RDMA usage instead of NVLink
    explicitly_destroy=True,
    allow_mnnvl=False,
)

print(f"Rank {rank}: DeepEP buffer created (RDMA forced via USE_MNNVL=0)")

# Create input data
x = torch.randn((batch_size, hidden), dtype=torch.bfloat16, device=f"cuda:{rank}")

# Create routing on rank 0 and broadcast
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

# Calculate actual data size
actual_tokens_transferred = batch_size * top_k
data_size_mb = actual_tokens_transferred * hidden * 2 / 1024 / 1024  # bfloat16 = 2 bytes


# E2E MoE function with expert computation
def run_e2e_moe():
    # DISPATCH using low-latency API (correct signature from test_low_latency.py)
    packed_recv_x, packed_recv_count, handle, event, hook = buffer.low_latency_dispatch(
        x, topk_idx, batch_size, num_experts, use_fp8=False, async_finish=True, return_recv_hook=False
    )

    # Wait for async dispatch to complete
    event.current_stream_wait()

    # EXPERT COMPUTATION
    if rank == 0:
        # GPU 0: No experts, just pass through received data
        simulated_gemm_x = packed_recv_x.clone()
    else:
        # GPU 1: Apply expert models to received tokens
        simulated_gemm_x = torch.zeros_like(packed_recv_x)

        # Process each local expert
        experts_per_rank = num_experts // world_size
        local_expert_start = rank * experts_per_rank

        for local_expert_idx in range(experts_per_rank):
            recv_count = packed_recv_count[local_expert_idx]
            num_tokens_for_expert = recv_count.item()

            if num_tokens_for_expert > 0:
                global_expert_idx = local_expert_start + local_expert_idx
                expert_tokens = packed_recv_x[local_expert_idx, :num_tokens_for_expert]

                # Apply expert computation
                with torch.no_grad():
                    expert_output = expert_models[global_expert_idx](expert_tokens)

                simulated_gemm_x[local_expert_idx, :num_tokens_for_expert] = expert_output

        torch.cuda.synchronize(f"cuda:{rank}")

    # Synchronize before combine
    dist.barrier()

    # COMBINE using low-latency API (correct signature from test_low_latency.py)
    combined_x, event, hook = buffer.low_latency_combine(
        simulated_gemm_x, topk_idx, topk_weights, handle, async_finish=True, return_recv_hook=False
    )

    # Wait for async combine to complete
    event.current_stream_wait()

    return combined_x


_ = run_e2e_moe()

dist.barrier()

start_time = timeit.default_timer()
result = run_e2e_moe()
end_time = timeit.default_timer()

e2e_latency_ms = (end_time - start_time) * 1000

if rank == 0:
    throughput = data_size_mb * 2 / e2e_latency_ms

    print(f"Rank {rank}: === DeepEP RDMA E2E MoE Results ===")
    print(f"Rank {rank}: E2E Latency (with expert computation): {e2e_latency_ms:.2f} ms")
    print(f"Rank {rank}: Data size: {batch_size} tokens x {top_k} experts x {hidden} hidden x 2 bytes = {data_size_mb:.1f} MB")
    print(f"Rank {rank}: Throughput: {throughput:.1f} MB/ms")
    print(f"Rank {rank}: Result shape: {result.shape}")

try:
    buffer.destroy()
except:
    pass
try:
    dist.destroy_process_group()
except:
    pass
