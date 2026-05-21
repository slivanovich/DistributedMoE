# MoE and P2P vs Collectives — concise intro

## Introduction

Mixture-of-Experts (MoE) models route each token to a small subset of expert subnetworks, enabling high parameter counts at near-constant per-token compute. A gate selects top-k experts; activations are dispatched, processed, and gathered. This sparsity shifts the bottleneck from compute to communication, traditionally handled with collectives (e.g., all-to-all).

### Short talk intro

Slide 1. Good evening! My name is … I am doing my thesis with advisor Vasily Ershov. Today I present my work: fault‑tolerant inference for large‑scale MoE models.

Slide 2. A few words about the field.
- Modern large language models need a lot of GPU memory to make good text.
- MoE models also use a lot of memory, so for inference we often need many GPUs.
- This makes the system harder: we must place model weights on different GPUs, make GPUs communicate, and route user requests.

What is MoE.
- Instead of one big dense layer in the model blocks (about 65–75% of LLM weights), MoE has many small experts (dense layers).
- Experts hold most of the parameters (about 85–95% or more).
- For each token, only some experts work. We use fewer weights in each forward pass, and text quality stays the same or better.

Slide 3. How we do inference today.
- We use distributed inference (for example, ideas from the DistServe paper).
- We cannot keep all experts on one GPU. We do not know in advance which experts we will need next.
- So we use Expert Parallelism and collective communication to move activations and results.

Why this is a problem.
- This plan is static. It does not work well when demand for experts is uneven.
- It is not fault‑tolerant. With collectives, if one GPU with any expert fails, the whole pipeline can stop, and we often must rebuild or restart the system.

Our idea.
- Replace collectives with peer‑to‑peer (one‑to‑one) transfers. We aim for similar speed and better fault tolerance.

### Project goals and objectives

Goals
- Deliver fault‑tolerant, large‑scale MoE inference that matches collective‑based throughput using P2P communication.
- Improve resilience to stragglers and GPU failures.
- Provide a clean, modular stack for MoE orchestration, expert execution, and high‑performance tensor transfer.

Objectives
- Implement P2P‑based dispatch/combine with per‑peer flow control (credits/backpressure), batching, and topology‑aware scheduling.
- Centralize asynchronous control paths and continuous liveness monitoring, with fast abort/retry/reroute.
- Optimize Expert Blocks and buffer pools to overlap routing, transfer, and compute for high link and SM utilization.
- Establish a rigorous baseline using collectives (all‑to‑all) for performance comparison.
- Benchmark across NVLink/InfiniBand: throughput, step time, token‑level latency percentiles, and achieved link/memory bandwidth.
- Validate fault tolerance: inject simulated GPU failures and measure their impact on generation latency and quality.

### Evaluation plan and baseline

We will evaluate with these metrics:
- Performance: get as close as possible to static LLM inference. Stats: end‑to‑end MoE layer latency, end‑to‑end latency, memory transfer throughput, etc.
- Fault tolerance: handle expert GPU failures and balance load. Stats: model quality and ability to generate when one or more GPUs fail.

Baseline
- We use the DeepEP library as a baseline. It is a distributed communication platform for MoE and Expert Parallelism. It gives fast, low‑latency GPU communication. It supports dispatch and combine operations in MoE.

### Static (classic) MoE inference with collectives and Expert Parallelism

The conventional deployment pattern for MoE uses Expert Parallelism (EP) with collective communication, typically all-to-all, inside an EP group:

- Expert placement: the global expert set is statically partitioned across EP ranks; each rank hosts a shard subset of experts with their weights resident in device memory.
- Routing and bucketing: the gate computes top-k experts per token; tokens are bucketed by destination rank/expert.
- Collective exchange (dispatch): an all-to-all ships token activations from each source rank to the ranks that host the selected experts.
- Local expert compute: each destination rank runs its local experts over the received token batches, commonly using grouped/batched GEMMs for high utilization.
- Collective exchange (combine): a second all-to-all returns expert outputs to the tokens’ owning ranks, followed by gating combine on the origin.

Characteristics and trade-offs:
- Deterministic phases: two collectives bracket the compute, simplifying scheduling and accounting but introducing global synchronization points.
- Load balance sensitivity: uneven routing increases per-rank variance; systems rely on auxiliary balancing losses, capacity limits, or token dropping/padding to bound tail latency.
- Performance model (back-of-envelope): step time ≈ 2 × Talltoall(payloadEP) + Tcompute. As scale grows, collective latency and bandwidth often dominate unless expert batches are large and well-packed.
- Failure domain: collectives are fragile to any single-rank failure or straggler; the whole operation stalls or aborts.

## P2P vs Collectives — hypothesis

Replacing collectives with peer-to-peer (P2P, one-to-one) can match throughput and improve fault tolerance.
- Throughput parity: with sufficient parallel channels, batching, and topology-aware scheduling, P2P saturates links similarly to all-to-all while avoiding collective coordination costs.
- Latency robustness: P2P removes global synchronization; stragglers or partial failures do not stall the whole group.
- Fault isolation: failures localize to affected peers; retries, rerouting, or expert reassignments proceed without tearing down a collective.
- Flow control: credit/backpressure on per-peer queues prevents buffer bloat and head-of-line blocking across the system.

Practical implications in this codebase:
- P2P data movement and control paths are encapsulated in the Tensor Transfer Engine: [src/python/TensorTransferEngine/TensorTransferEngine.py](src/python/TensorTransferEngine/TensorTransferEngine.py) and implementations [src/python/TensorTransferEngine/MCTETensorTransferEngine.py](src/python/TensorTransferEngine/MCTETensorTransferEngine.py), [src/python/TensorTransferEngine/NIXLTensorTransferEngine.py](src/python/TensorTransferEngine/NIXLTensorTransferEngine.py).
- Centralized async handling within the MoE side aligns with P2P: see [src/python/MoE/DistMoEBlock.py](src/python/MoE/DistMoEBlock.py) and EB orchestration in [src/python/MoE/DistExpertsBlock.py](src/python/MoE/DistExpertsBlock.py).

What to measure to validate the hypothesis:
- End-to-end step time, token-dispatch latency percentiles, and achieved bandwidth (NVLink/IB).
- Robustness under injected stragglers/failures: time-to-recovery and work preserved.

Relevant artifacts for empirical comparison are in SAFE datasets (e.g., throughput/latency CSVs and plots under [SAFE/data/pull-push_schema](SAFE/data/pull-push_schema) and [SAFE/data/push-push_schema](SAFE/data/push-push_schema)).
