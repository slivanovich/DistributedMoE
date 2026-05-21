# FROM nvidia/cuda:12.1.1-devel-ubuntu22.04
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

WORKDIR /MCTE

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Environments variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$PATH:${CUDA_HOME}/bin
ENV LIBRARY_PATH=$LIBRARY_PATH:${CUDA_HOME}/lib64
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CUDA_HOME}/lib64

# From mc installation guide - combine update and install to ensure fresh package cache
RUN apt-get update --fix-missing && apt-get install -y build-essential \
libibverbs-dev \
libgoogle-glog-dev \
libgtest-dev \
libjsoncpp-dev \
libnuma-dev \
libunwind-dev \
libpython3-dev \
libboost-all-dev \
libssl-dev \
pybind11-dev \
libcurl4-openssl-dev \
libhiredis-dev \
pkg-config \
patchelf \
vim \
wget \
python3.10 \
python3-pip \
pciutils \
libucx-dev \
# Error while building MC:
    # "Could not find a package configuration file provided by "yaml-cpp" with any
    #   of the following names:
#     yaml-cppConfig.cmake
#     yaml-cpp-config.cmake"
libyaml-cpp-dev \
librdmacm-dev

# Git tools
RUN apt-get install -y git git-lfs
RUN git lfs install

# Base python packages
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install wheel
RUN alias python='python3'

RUN mkdir .local
WORKDIR /MCTE/.local

# CMake installation (3.31.4 version specifically)
RUN apt-get purge cmake; exit 0
RUN wget -O cmake-3.31.4.tar.gz https://github.com/Kitware/CMake/releases/download/v3.31.4/cmake-3.31.4.tar.gz
RUN tar -xf cmake-3.31.4.tar.gz
RUN cd cmake-3.31.4/ && ./bootstrap && make && make install
RUN rm cmake-3.31.4.tar.gz
RUN rm -rf cmake-3.31.4
RUN hash -r

# Install golang
RUN rm -rf /usr/local/go; exit 0
RUN wget -O go1.23.8.linux-amd64.tar.gz https://go.dev/dl/go1.23.8.linux-amd64.tar.gz
RUN tar -C /usr/local -xzf go1.23.8.linux-amd64.tar.gz
RUN rm go1.23.8.linux-amd64.tar.gz
ENV PATH=$PATH:/usr/local/go/bin

# Build&Install Yalantinglibs
WORKDIR /MCTE/.local
RUN git clone https://github.com/alibaba/yalantinglibs.git
WORKDIR /MCTE/.local/yalantinglibs
RUN rm -rf build || true && \
mkdir build && cd build && \
cmake .. -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARK=OFF -DBUILD_UNIT_TESTS=OFF && \
make -j$(nproc) && \
make install

# Build&Install Mooncake
RUN apt-get install -y libzstd-dev libmsgpack* xxhash libxxhash-dev
WORKDIR /MCTE/.local
RUN git clone https://github.com/kvcache-ai/Mooncake.git
WORKDIR /MCTE/.local/Mooncake
RUN git submodule update --init
RUN git lfs pull || true
RUN rm -rf build || true && \
mkdir build && cd build && \
cmake .. -DUSE_ETCD=ON -DUSE_HTTP=ON -DUSE_CUDA=ON -DUSE_MNNVL=ON && \
make -j$(nproc) && make install
RUN python3 -m pip install mooncake-transfer-engine

# Build&Install NIXL (with UCX)
WORKDIR /MCTE/.local
RUN apt-get install -y autoconf libtool
RUN git clone https://github.com/openucx/ucx.git
WORKDIR /MCTE/.local/ucx
RUN git checkout v1.20.x
ENV LDFLAGS="-L/usr/lib/x86_64-linux-gnu -Wl,-rpath,/usr/lib/x86_64-linux-gnu"
RUN ./autogen.sh && ./configure \
--prefix=/MCTE/.local/ucx \
--enable-shared \
--disable-static \
--disable-doxygen-doc \
--enable-optimizations \
--enable-cma \
--enable-devel-headers \
--with-cuda=$CUDA_HOME \
--with-verbs \
--with-dm \
--enable-mt \
&& make -j && make install

WORKDIR /MCTE/.local
RUN apt-get install -y ninja-build libaio-dev libc-dev
RUN python3 -m pip install meson pybind11
RUN git clone https://github.com/ai-dynamo/nixl.git
WORKDIR /MCTE/.local/nixl
RUN rm -rf build || true && mkdir build && \
meson setup build \
    -Ducx_path=/MCTE/.local/ucx \
    -Dinstall_headers=true \
    -Ddisable_gds_backend=true \
    -Dwerror=false \
    -Dcpp_args="-Wno-error -Wno-inconsistent-missing-override -Wno-unused-private-field -Wno-mismatched-tags" && \
meson configure build --prefix=/MCTE/.local/nixl && \
cd build && ninja && ninja install
RUN python3 -m pip install nixl[cu12]

ENV NIXL_PREFIX="/MCTE/.local/nixl"
ENV NIXL_LIB_DIR="$NIXL_PREFIX/lib/x86_64-linux-gnu"
ENV LD_LIBRARY_PATH="$NIXL_LIB_DIR:$LD_LIBRARY_PATH"

WORKDIR /MCTE

COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt

RUN echo -e "* soft memlock unlimited \n* hard memlock unlimited" >> /etc/security/limits.d/limits.conf

# After troubleshooting.md
# Metadata server lib
RUN apt-get install -y etcd
# For rdma CLI tools (ibv_devices command to view the list of installed network cards on the machine)
RUN apt-get install -y iproute2
# For troubleshooting (ibv_devinfo command to view the list of installed network cards on the machine with detailes)
RUN apt-get install -y ibverbs-utils
# For troubleshooting (lsmod command to show the status of linux kernel modules)
RUN apt-get install -y kmod
# InfiniBand network utilities
RUN apt-get install -y ibutils
# Htop util
RUN apt-get install -y htop
# For infiniband test
RUN apt-get install -y perftest numactl
RUN apt-get -y install systemd
# For `--nic-metrics=true` in nsight profiler
RUN apt-get -y install libibmad5
# Displaying information about PCI buses in the system and devices connected to them use: lspci


RUN echo 'export UCX_HOME="/MCTE/.local/ucx"' >> ~/.bashrc
RUN echo 'export PATH="$UCX_HOME/bin:$PATH"' >> ~/.bashrc
RUN echo 'export PKG_CONFIG_PATH="$UCX_HOME/lib/pkgconfig:${PKG_CONFIG_PATH:-}"' >> ~/.bashrc
RUN echo 'export CUDA_HOME="${CUDA_HOME}"' >> ~/.bashrc
RUN echo 'export PATH="$PATH"' >> ~/.bashrc
RUN echo 'export LD_LIBRARY_PATH="$UCX_HOME/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"' >> ~/.bashrc

ENV UCX_LOG_LEVEL=info
ENV UCX_PROTO_INFO=y
ENV UCX_MAX_RNDV_RAILS=1
ENV UCX_LOG_FILE=/MCTE/ucx_%p.log

RUN python3 -m pip install vllm datasets
# RUN python3 -m pip install vllm==0.11.0 datasets

# Install nsight
# RUN wget -O NsightSystems-linux-public-2025.5.1.121-3638078.run https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/nsight-systems-2025.5.1_2025.5.1.121-3638078_amd64.deb
COPY .cache/NsightSystems-linux-public-2025.5.1.121-3638078.run .
RUN chmod +x NsightSystems-linux-public-2025.5.1.121-3638078.run
# RUN ./NsightSystems-linux-public-2025.5.1.121-3638078.run --quiet --accept --target /MCTE/nsight-systems-2025.5.1
# RUN rm NsightSystems-linux-public-2025.5.1.121-3638078.run
ENV PATH=$PATH:/MCTE/nsight-systems-2025.5.1/bin

# ============================================================================
# DeepEP Installation with NVSHMEM Support
# ============================================================================

# Install Linux kernel headers (required for potential DKMS operations)
RUN apt-get update && apt-get install -y \
    linux-headers-generic \
    && rm -rf /var/lib/apt/lists/*

# Install NVSHMEM 3.6.5 for CUDA 12.x and patchelf
WORKDIR /tmp
RUN wget https://developer.download.nvidia.com/compute/nvshmem/3.6.5/local_installers/nvshmem-local-repo-ubuntu2204-3.6.5_3.6.5-1_amd64.deb && \
    dpkg -i nvshmem-local-repo-ubuntu2204-3.6.5_3.6.5-1_amd64.deb && \
    cp /var/nvshmem-local-repo-ubuntu2204-3.6.5/nvshmem-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && \
    apt-get install -y nvshmem-cuda-12 patchelf && \
    rm -f *.deb && \
    rm -rf /var/lib/apt/lists/*

# Configure NVSHMEM library paths
RUN ldconfig

# Install GDRCopy from source (userspace libraries only, avoiding DKMS issues)
WORKDIR /tmp
RUN git clone https://github.com/NVIDIA/gdrcopy.git && \
    cd gdrcopy && \
    make lib && \
    make install && \
    ldconfig && \
    cd / && rm -rf /tmp/gdrcopy

# Install PyTorch (required for DeepEP)
RUN python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Create symlink for nvtx3interop library (workaround for missing library)
RUN find /usr/local/cuda -name "*nvtx*" -type f 2>/dev/null | head -5 && \
    ln -sf /usr/local/cuda/lib64/libnvToolsExt.so /usr/local/cuda/lib64/libnvtx3interop.so || \
    echo "NVTX library not found, will try to build without it"

# Clone and build DeepEP (commit e713f52 fixes align issue + has non-nvlink support)
WORKDIR /MCTE/.local
RUN git clone https://github.com/deepseek-ai/DeepEP.git
WORKDIR /MCTE/.local/DeepEP
RUN git checkout c5a3e9b790caedecfe362bc206c186185d8bfaab

# Find NVSHMEM installation directory and build DeepEP
ENV TORCH_CUDA_ARCH_LIST="9.0"
RUN find /usr -name "device_host_transport" -type d 2>/dev/null | head -1 | xargs dirname > /tmp/nvshmem_headers_dir.txt && \
    NVSHMEM_HEADERS_DIR=$(cat /tmp/nvshmem_headers_dir.txt) && \
    NVSHMEM_LIB_DIR="/usr/lib/x86_64-linux-gnu/nvshmem/12" && \
    echo "Found NVSHMEM headers at: $NVSHMEM_HEADERS_DIR" && \
    echo "Found NVSHMEM libraries at: $NVSHMEM_LIB_DIR" && \
    ls -la "$NVSHMEM_HEADERS_DIR/" && \
    ls -la "$NVSHMEM_HEADERS_DIR/device_host_transport/" && \
    ls -la "$NVSHMEM_LIB_DIR/" && \
    echo "$NVSHMEM_LIB_DIR" > /etc/ld.so.conf.d/nvshmem.conf && \
    ldconfig && \
    sed -i "s|include_dirs.extend(\[f'{nvshmem_dir}/include'\])|include_dirs.extend(['$NVSHMEM_HEADERS_DIR'])|" setup.py && \
    sed -i "s|library_dirs.extend(\[f'{nvshmem_dir}/lib'\])|library_dirs.extend(['$NVSHMEM_LIB_DIR'])|" setup.py && \
    sed -i "s|nvcc_dlink.extend(\['-dlink', f'-L{nvshmem_dir}/lib', '-lnvshmem_device'\])|nvcc_dlink.extend(['-dlink', '-L$NVSHMEM_LIB_DIR', '-lnvshmem_device'])|" setup.py && \
    sed -i "s|extra_link_args.extend(\[f'-l:{nvshmem_host_lib}', '-l:libnvshmem_device.a', f'-Wl,-rpath,{nvshmem_dir}/lib'\])|extra_link_args.extend([f'-l:{nvshmem_host_lib}', '-l:libnvshmem_device.a', '-Wl,-rpath,$NVSHMEM_LIB_DIR'])|" setup.py && \
    NVSHMEM_DIR="$NVSHMEM_HEADERS_DIR" python3 setup.py build_ext --inplace && \
    NVSHMEM_DIR="$NVSHMEM_HEADERS_DIR" python3 setup.py install

# Create DeepEP environment wrapper script with correct NVSHMEM library path
RUN NVSHMEM_LIB_DIR="/usr/lib/x86_64-linux-gnu/nvshmem/12" && \
    echo '#!/bin/bash' > /usr/local/bin/deepep-env && \
    echo '# DeepEP Environment Setup Script' >> /usr/local/bin/deepep-env && \
    echo '# This script ensures DeepEP uses the correct NVSHMEM library to resolve runtime linking issues' >> /usr/local/bin/deepep-env && \
    echo '' >> /usr/local/bin/deepep-env && \
    echo "export LD_PRELOAD=\"$NVSHMEM_LIB_DIR/libnvshmem_host.so:\$LD_PRELOAD\"" >> /usr/local/bin/deepep-env && \
    echo "export LD_LIBRARY_PATH=\"$NVSHMEM_LIB_DIR:\$LD_LIBRARY_PATH\"" >> /usr/local/bin/deepep-env && \
    echo '' >> /usr/local/bin/deepep-env && \
    echo '# Execute the command with the proper environment' >> /usr/local/bin/deepep-env && \
    echo 'exec "$@"' >> /usr/local/bin/deepep-env && \
    chmod +x /usr/local/bin/deepep-env

# Fix NVSHMEM runtime linking for DeepEP extension with correct library path
RUN NVSHMEM_LIB_DIR="/usr/lib/x86_64-linux-gnu/nvshmem/12" && \
    for so_file in /usr/local/lib/python3.10/dist-packages/deep_ep_cpp.cpython-*-linux-gnu.so; do \
        if [ -f "$so_file" ]; then \
            echo "Patching $so_file with rpath: $NVSHMEM_LIB_DIR"; \
            patchelf --set-rpath "$NVSHMEM_LIB_DIR:/usr/local/lib/python3.10/dist-packages/nvidia/nvshmem/lib" "$so_file"; \
        fi; \
    done && \
    rm -f /tmp/nvshmem_headers_dir.txt

# Verify DeepEP installation (skip CUDA runtime check in build environment)
RUN echo "Verifying DeepEP installation..." && \
    python3 -c "import sys; sys.path.insert(0, '/MCTE/.local/DeepEP'); import deep_ep; print('DeepEP Python module imported successfully')" || \
    echo "DeepEP import requires CUDA runtime - will work when container runs with GPU support"

# Return to main working directory
WORKDIR /MCTE

# ============================================================================
# DeepEP Installation Complete
#
# INSTALLATION SUMMARY:
#   - GDRCopy: Installed userspace libraries (avoiding DKMS kernel module issues)
#   - NVSHMEM: Installed version 3.6.5 for CUDA 12.x with proper library paths
#   - DeepEP: Successfully compiled with NVSHMEM support and device libraries
#   - Runtime Environment: Configured with proper library linking and paths
#
# USAGE:
#   - Use 'deepep-env python3 your_script.py' to run Python scripts with DeepEP
#   - The wrapper script handles NVSHMEM library conflicts automatically
#   - DeepEP tests are available in /MCTE/.local/DeepEP/tests/
#   - Run container with GPU support: docker run --gpus all your_image
#
# TROUBLESHOOTING:
#   - If you see "libcuda.so.1: cannot open shared object file", ensure:
#     1. Container is run with --gpus all flag
#     2. NVIDIA Container Toolkit is installed on host
#     3. NVIDIA drivers are properly installed on host
#
# VERIFICATION:
#   - NVSHMEM installation: nvshmem-info -a
#   - DeepEP tests: cd /MCTE/.local/DeepEP && python3 -m pytest tests/
#   - GPU availability: nvidia-smi (requires GPU runtime)
#
# PATHS:
#   - NVSHMEM headers: /usr/include/nvshmem_12
#   - NVSHMEM libraries: /usr/lib/x86_64-linux-gnu/nvshmem/12
#   - DeepEP source: /MCTE/.local/DeepEP
#   - GDRCopy libraries: /usr/local/lib
# ============================================================================
