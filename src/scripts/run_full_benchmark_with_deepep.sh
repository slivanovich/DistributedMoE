#!/bin/bash

# Usage: "bash src/scripts/run_full_benchmark_with_deepep.sh [mcte/nixl] [fp16/fp32/bf16] [host_gpu] [remote_gpu] [rdma/nvlink] [--bench-deep-ep] [--bench-dist-moe]"
# Example: "bash src/scripts/run_full_benchmark_with_deepep.sh nixl bf16 0 1 rdma --bench-deep-ep --bench-dist-moe"

set -e  # Exit on any error

# Parse command line arguments
BACKEND=${1:-nixl}
DTYPE=${2:-bf16}
HOST_GPU=${3:-0}
REMOTE_GPU=${4:-1}
TRANSFER_PROTOCOL=${5:-rdma}
BENCH_DIST_MOE=false
BENCH_DEEP_EP=false

# Check for --bench-dist-moe flag
for arg in "$@"; do
    if [ "$arg" = "--bench-dist-moe" ]; then
        BENCH_DIST_MOE=true
        break
    fi
done

# Check for --bench-deep-ep flag
for arg in "$@"; do
    if [ "$arg" = "--bench-deep-ep" ]; then
        BENCH_DEEP_EP=true
        break
    fi
done

echo "=========================================="
echo "Full MoE Benchmark with DeepEP Baseline"
echo "=========================================="

# SEQUENCE_LENGTH=200
BATCH_SIZES="1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000,3200,3400,3600,3800,4000,4200,4400,4600,4800,5000,5200,5400,5600,5800,6000,6200,6400,6600,6800,7000,7200,7400,7600,7800,8000,8200,8400,8600,8800,9000"
OUTPUT_DIR="/MCTE/src/python/MoE/benchmarks/data/comprehensive_benchmark"
HOST_DEVICE="cuda:$HOST_GPU"
REMOTE_DEVICE="cuda:$REMOTE_GPU"
WARMUP_RUNS=16
PERF_RUNS=32

echo "Configuration:"
echo "- Backend: $BACKEND"
echo "- Host Device: $HOST_DEVICE"
echo "- Remote Device: $REMOTE_DEVICE"
echo "- Data Type: $DTYPE"
echo "- Transfer Protocol: $TRANSFER_PROTOCOL"
echo "- Warmup Runs: $WARMUP_RUNS"
echo "- Performance Runs: $PERF_RUNS"
echo "- Batch Sizes: $BATCH_SIZES"
echo "- Output Directory: $OUTPUT_DIR"
echo "- Run DistMoE Benchmark: $BENCH_DIST_MOE"
echo ""

export CUDA_VISIBLE_DEVICES=${HOST_GPU},${REMOTE_GPU}

pushd /MCTE/src/python

# Step 1: Run comprehensive DistMoE benchmark (only if --bench-dist-moe flag is provided)
if [ "$BENCH_DIST_MOE" = true ]; then
    echo "=========================================="
    echo "Step 1: Running Comprehensive DistMoE Benchmark"
    echo "=========================================="

    MC_LOG_LEVEL=WARNING python3 -m MoE.benchmarks.comprehensive_benchmark \
        --backend $BACKEND \
        --host_device $HOST_DEVICE \
        --remote_device $REMOTE_DEVICE \
        --dtype $DTYPE \
        --transfer_protocol $TRANSFER_PROTOCOL \
        --warmup_runs $WARMUP_RUNS \
        --perf_runs $PERF_RUNS \
        --output_dir $OUTPUT_DIR \
        --batch_sizes $BATCH_SIZES

    if [ $? -eq 0 ]; then
        echo "✓ Comprehensive DistMoE benchmark completed successfully"
    else
        echo "✗ Comprehensive DistMoE benchmark failed"
        exit 1
    fi

    echo ""
else
    echo "Skipping DistMoE benchmark (use --bench-dist-moe to enable)"
    echo ""
fi

# Step 2: Run DeepEP baseline benchmark
if [ "$BENCH_DEEP_EP" = true ]; then
    echo "=========================================="
    echo "Step 2: Running DeepEP Baseline Benchmark"
    echo "=========================================="

    # Convert comma-separated batch sizes to space-separated for DeepEP script
    DEEPEP_BATCH_SIZES=$(echo $BATCH_SIZES | tr ',' ' ')

    deepep-env torchrun --nproc_per_node=2 \
        --master_addr=localhost \
        --master_port=12355 \
        /MCTE/src/python/MoE/benchmarks/deepep_baseline_benchmark.py \
        --batch-sizes $DEEPEP_BATCH_SIZES \
        --num-warmup 2 \
        --num-runs 5 \
        --output-dir $OUTPUT_DIR/$TRANSFER_PROTOCOL/deepep/csv

    if [ $? -eq 0 ]; then
        echo "✓ DeepEP baseline benchmark completed successfully"
    else
        echo "✗ DeepEP baseline benchmark failed"
        exit 1
    fi
    echo ""
else
    echo "Skipping DeepEP benchmark (use --bench-deep-ep to enable)"
    echo ""
fi

# Step 3: Generate comprehensive graphs with DeepEP baseline
echo "=========================================="
echo "Step 3: Generating Comprehensive Graphs with DeepEP Baseline"
echo "=========================================="

python3 -m MoE.benchmarks.comprehensive_benchmark \
    --backend $BACKEND \
    --graph-only \
    --transfer_protocol $TRANSFER_PROTOCOL \
    --output_dir $OUTPUT_DIR

if [ $? -eq 0 ]; then
    echo "✓ Graph generation completed successfully"
else
    echo "✗ Graph generation failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "All benchmarks completed successfully!"
echo "=========================================="
