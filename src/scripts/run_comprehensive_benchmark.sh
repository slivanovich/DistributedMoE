#!/bin/bash

# Usage: "bash src/scripts/run_comprehensive_benchmark.sh [mcte/nixl] [fp16/fp32/bf16] [host_gpu] [remote_gpu] [rdma/nvlink]"
# Example: "bash src/scripts/run_comprehensive_benchmark.sh nixl fp16 0 1 rdma"

BACKEND=${1:-nixl}
DTYPE=${2:-fp16}
HOST_GPU=${3:-0}
REMOTE_GPU=${4:-1}
TRANSFER_PROTOCOL=${5:-rdma}
OUTPUT_DIR="/MCTE/src/python/MoE/benchmarks/data/comprehensive_benchmark"
BATCH_SIZES="1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000,3200,3400,3600,3800,4000,4200,4400,4600,4800,5000,5200,5400,5600,5800,6000,6200,6400,6600,6800,7000,7200,7400,7600,7800,8000,8200,8400,8600,8800,9000"

export CUDA_VISIBLE_DEVICES=${HOST_GPU},${REMOTE_GPU}

pushd /MCTE/src/python

MC_LOG_LEVEL=WARNING python3 -m MoE.benchmarks.comprehensive_benchmark \
    --backend=${BACKEND} \
    --host_device=cuda:${HOST_GPU} \
    --remote_device=cuda:${REMOTE_GPU} \
    --dtype=${DTYPE} \
    --transfer_protocol=${TRANSFER_PROTOCOL} \
    --output_dir=${OUTPUT_DIR} \
    --batch_sizes=${BATCH_SIZES}

popd