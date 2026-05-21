#!/bin/bash

# Usage: bash src/scripts/connect_to_docker.sh "<cmd>"

docker exec -it skuralenok-dist-moe /bin/bash -c "$1"
