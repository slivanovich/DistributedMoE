 #!/bin/bash

# Usage: bash src/scripts/run_all_tests.sh


echo "---------------------------------------------------------------------------------------"
echo -e "Infiniband avaibility and throughput test.\n"
bash /MCTE/src/scripts/infiniband_test.sh
echo "---------------------------------------------------------------------------------------"
# echo -e "UCX (RDMA) avaibility and throughput test.\n"
# source ~/.bashrc
# UCX_WARN_UNUSED_ENV_VARS=n CUDA_VISIBLE_DEVICES=6 UCX_NET_DEVICES=mlx5_2:1 UCX_TLS=rc,cuda_copy ucx_perftest -t tag_bw -m cuda -s 10000000 -n 10 -p 9999 & \
# UCX_WARN_UNUSED_ENV_VARS=n CUDA_VISIBLE_DEVICES=7 UCX_NET_DEVICES=mlx5_3:1 UCX_TLS=rc,cuda_copy ucx_perftest `hostname` -t tag_bw -m cuda -s 100000000 -n 10 -p 9999
# echo -e "---------------------------------------------------------------------------------------\n"
# echo -e "UCX (NVLink) avaibility and throughput test.\n"
# UCX_WARN_UNUSED_ENV_VARS=n CUDA_VISIBLE_DEVICES=6 UCX_TLS=cuda_copy,cuda_ipc,tcp ucx_perftest -t tag_bw -m cuda -s 10000000 -n 10 -p 9999 & \
# UCX_WARN_UNUSED_ENV_VARS=n CUDA_VISIBLE_DEVICES=7 UCX_TLS=cuda_copy,cuda_ipc,tcp ucx_perftest `hostname` -t tag_bw -m cuda -s 100000000 -n 10 -p 9999
echo -e "---------------------------------------------------------------------------------------\n"

export CUDA_VISIBLE_DEVICES=6,7

pushd /MCTE/src/python

# -s flag for tests logs (prints, etc.)
python3 -m TensorTransfer.tests.tt_mcte_test --gpu_id_host=0 --gpu_id_remote=1
python3 -m TensorTransfer.tests.tt_nixl_test --gpu_id_host=0 --gpu_id_remote=1

python3 -m MoE.tests.dist_moe_test --backend=mcte --precision=fp16 --gpu_id_host=0 --gpu_id_remote=1 --schema=push-push -s
python3 -m MoE.tests.dist_moe_test --backend=nixl --precision=fp16 --gpu_id_host=0 --gpu_id_remote=1 --schema=push-push -s

python3 -m MoE.tests.dist_moe_test --backend=mcte --precision=fp16 --gpu_id_host=0 --gpu_id_remote=1 --schema=pull-push -s
python3 -m MoE.tests.dist_moe_test --backend=nixl --precision=fp16 --gpu_id_host=0 --gpu_id_remote=1 --schema=pull-push -s

popd
