#!/bin/bash

# Resolve the real user's home dir even when invoked via sudo
if [ -n "${SUDO_USER:-}" ]; then
    REAL_HOME=$(getent passwd "$SUDO_USER" | cut -d: -f6)
else
    REAL_HOME="$HOME"
fi
export HOME_PATH="${REAL_HOME}/skuralenok"

echo "HOME_PATH: $HOME_PATH"

echo "Stopping and removing old container..."
(docker container stop skuralenok-dist-moe 2> /dev/null || true) && (docker container rm skuralenok-dist-moe 2> /dev/null || true)

echo "Creating (if they are not existing yet) volumes..."
docker volume rm skuralenok_models skuralenok_mcte_src skuralenok_nsight
docker volume create --driver local -o o=bind -o type=none -o device="${HOME_PATH}/models" skuralenok_models
docker volume create --driver local -o o=bind -o type=none -o device="${HOME_PATH}/MCTE/src" skuralenok_mcte_src
# docker volume create --driver local -o o=bind -o type=none -o device="${HOME_PATH}/nsight-systems-2025.5.1" skuralenok_nsight

echo "Building docker..."
docker build --network=host -t skuralenok/dist-moe-img .

echo "Running docker..."
# -v "skuralenok_nsight:/MCTE/nsight-systems-2025.5.1"
# --device=/dev/infiniband/uverbs0 \
# --ulimit memlock=-1 \ -- это важная тема, без этого почему то падает создание TE на IB (можно чекать в целом доступность IB скриптом infiniband_test.sh)
docker run \
    --privileged=true \
    --network=host \
    --security-opt seccomp=docker_profile.json \
    --detach \
    --gpus all \
    --shm-size=32g \
    --ulimit memlock=-1 \
    --ipc=host \
    --tty \
    -v "skuralenok_models:/MCTE/models" \
    -v "skuralenok_mcte_src:/MCTE/src" \
    --name=skuralenok-dist-moe skuralenok/dist-moe-img

# echo "Connecting to the http metadata server (only for MoonCake TE)..."
# sudo bash connect_to_docker.sh "cd src/http-metadata-server/ && bash run_http_metadata_server.sh"
