 #!/bin/bash

# Usage: bash src/scripts/run_tt_tests.sh

export CUDA_VISIBLE_DEVICES=0,1

pushd /MCTE/src/python

python3 -m TensorTransfer.tests.tt_mcte_test --gpu_id_host=0 --gpu_id_remote=1 -s
python3 -m TensorTransfer.tests.tt_nixl_test --gpu_id_host=0 --gpu_id_remote=1 -s

popd