#!/bin/bash

# Usage: "bash src/scripts/run_fault_tolerance_benchmark.sh [backend] [dtype] [host_gpu] [expert_gpu1] [expert_gpu2] [transfer_protocol] [target_rps] [duration]"
# Example: "bash src/scripts/run_fault_tolerance_benchmark.sh nixl fp16 0 1 2 rdma 10.0 30"

BACKEND=${1:-nixl}
DTYPE=${2:-fp16}
HOST_GPU=${3:-0}
EXPERT_GPU1=${4:-1}
EXPERT_GPU2=${5:-2}
TRANSFER_PROTOCOL=${6:-rdma}
TARGET_RPS=${7:-50.0}
DURATION=${8:-30}
OUTPUT_DIR="/MCTE/src/python/MoE/benchmarks/data/fault_tolerance"

export CUDA_VISIBLE_DEVICES=${HOST_GPU},${EXPERT_GPU1},${EXPERT_GPU2}

pushd /MCTE/src/python

MC_LOG_LEVEL=WARNING python3 -m MoE.benchmarks.fault_tolerance_benchmark \
    --backend=${BACKEND} \
    --dtype=${DTYPE} \
    --host_device=cuda:${HOST_GPU} \
    --expert_device1=cuda:${EXPERT_GPU1} \
    --expert_device2=cuda:${EXPERT_GPU2} \
    --transfer_protocol=${TRANSFER_PROTOCOL} \
    --target_rps=${TARGET_RPS} \
    --duration=${DURATION} \
    --output-dir=${OUTPUT_DIR}

popd