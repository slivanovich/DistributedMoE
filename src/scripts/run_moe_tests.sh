 #!/bin/bash

# Usage: "bash src/scripts/run_moe_tests.sh [mcte/nixl]"

export CUDA_VISIBLE_DEVICES=0,1

pushd /MCTE/src/python

# nsys profile \
#  -o DistMoE_${1}_test \
#  --force-overwrite true \
#  --trace nvtx,cublas,cuda,cudnn,ucx \
#  --nic-metrics=true \
#  python3 -m MoE.tests.dist_moe_test --backend=$1 --precision=fp16 --gpu_id_host=0 --gpu_id_remote=1 -s
MC_LOG_LEVEL=WARNING python3 -m MoE.tests.dist_moe_test --backend=$1 --precision=fp16 --gpu_id_host=0 --gpu_id_remote=1 -s

popd