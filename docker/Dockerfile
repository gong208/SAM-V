# SAM-V dependency image: Python 3.10 + the pinned requirements.txt.
# Code is bind-mounted at runtime (docker/run.sh); docker/Dockerfile.deploy bakes it in.
#
# Build from the REPO ROOT so requirements.txt is in the context:
#   bash docker/build.sh
#
# Base image: torch's cu121 wheels bundle their own CUDA 12.1 runtime, so the
# base's CUDA 11.7 is irrelevant at runtime -- only the host NVIDIA driver must
# support CUDA 12.1. Ubuntu 22.04 ships Python 3.10.12, matching the pins.
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

# Optional build-time proxy. Nothing is hardcoded: build.sh forwards these only
# if they are set in your environment, and they are cleared from the final image.
ARG HTTP_PROXY=
ARG HTTPS_PROXY=
ARG NO_PROXY=
ENV http_proxy=$HTTP_PROXY https_proxy=$HTTPS_PROXY no_proxy=$NO_PROXY \
    HTTP_PROXY=$HTTP_PROXY HTTPS_PROXY=$HTTPS_PROXY NO_PROXY=$NO_PROXY

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# Python 3.10, git (submodules), and the runtime libs opencv-python needs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-dev \
        git libgl1 libglib2.0-0 ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# The SAME pins as a native install -- one source of truth, validated by
# tools/check_clean_install.sh. torch==2.3.1 pulls its CUDA 12.1 runtime wheels.
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install -r /tmp/requirements.txt

# Guard against timm 1.0.23, whose broken timm.models.layers shim kills sam-hq's
# imports (see requirements.txt). Fails the build rather than the first run.
RUN python -c "import timm; from timm.models.layers import DropPath, to_2tuple, trunc_normal_; print('timm', timm.__version__, 'OK')"

ENV http_proxy= https_proxy= no_proxy= HTTP_PROXY= HTTPS_PROXY= NO_PROXY=
ENV PYTHONPATH=/workspace:/workspace/submodules/sam-hq:/workspace/submodules/vggt

WORKDIR /workspace
CMD ["/bin/bash"]
