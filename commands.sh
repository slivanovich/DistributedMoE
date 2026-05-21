#!/bin/bash

# For DistMoE bench launch ([NVLink/RDMA]x[1..expert_blocks_number]x[NONE/MCTE/NIXL]x[batches])
UCX_LOG_LEVEL=info
UCX_PROTO_INFO=y
UCX_MAX_RNDV_RAILS=1
UCX_LOG_FILE=/MCTE/ucx_%p.log
bash connect_to_docker.sh "export CUDA_VISIBLE_DEVICES=0,1,2,3,4 && cd src/python && python3 -m MoE.benchmarks.bench --EB=4 --host_gpu_id=0 --EB_gpu_ids=1,2,3,4 --number_of_experts=64 --top_k=8 --io_buffer_size=4000"

# For throughput TensorTransfer bench launch (via RDMA)
bash connect_to_docker.sh "export CUDA_VISIBLE_DEVICES=6 && cd src/python && python3 -m TensorTransfer.benchmarks.bench --mode=initiator --local_address=localhost:10000 --remote_address=localhost:20000 transfer_protocol=rdma --ib_devices_names=mlx5_2 --gpu_id=0"
bash connect_to_docker.sh "export CUDA_VISIBLE_DEVICES=7 && export MC_FORCE_MNNVL=False && cd src/python && python3 -m TensorTransfer.benchmarks.bench --mode=target --local_address=localhost:20000 --remote_address=localhost:10000 transfer_protocol=rdma --ib_devices_names=mlx5_3 --gpu_id=0"

# For throughput TensorTransfer bench launch (via NVLink)
bash connect_to_docker.sh "export CUDA_VISIBLE_DEVICES=6 && export MC_FORCE_MNNVL=True && cd src/python && python3 -m TensorTransfer.benchmarks.bench --mode=initiator --local_address=localhost:10000 --remote_address=localhost:20000 transfer_protocol=nvlink --gpu_id=0"
bash connect_to_docker.sh "export CUDA_VISIBLE_DEVICES=7 && export MC_FORCE_MNNVL=True && cd src/python && python3 -m TensorTransfer.benchmarks.bench --mode=target --local_address=localhost:20000 --remote_address=localhost:10000 transfer_protocol=nvlink --gpu_id=0"
