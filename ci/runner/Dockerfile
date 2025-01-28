# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Args used in FROM commands must come first
ARG FROM_IMAGE="rapidsai/ci-conda"
ARG CUDA_PKG_VER=12-0
ARG CUDA_SHORT_VER=12.5
ARG CUDA_VER=12.5.1
ARG LINUX_DISTRO=ubuntu
ARG LINUX_VER=22.04
ARG PROJ_NAME=morpheus
ARG PYTHON_VER=3.10
ARG REAL_ARCH=x86_64


# Configure the base docker img
FROM --platform=$TARGETPLATFORM ${FROM_IMAGE}:cuda${CUDA_VER}-${LINUX_DISTRO}${LINUX_VER}-py${PYTHON_VER} AS base

ARG PROJ_NAME
ARG CUDA_SHORT_VER

SHELL ["/bin/bash",  "-c"]

ENV REAL_ARCH=${REAL_ARCH}

# Create conda environment
COPY ./dependencies.yaml /tmp/conda/

# ============ build ==================
FROM base as build

# Add any build only dependencies here.
ARG CUDA_SHORT_VER
ARG PROJ_NAME
ARG PYTHON_VER
ARG REAL_ARCH

RUN rapids-dependency-file-generator \
      --config /tmp/conda/dependencies.yaml \
      --output conda \
      --file-key build \
      --matrix "cuda=${CUDA_SHORT_VER};arch=${REAL_ARCH};py=${PYTHON_VER}" > /tmp/conda/env.yaml && \
    CONDA_ALWAYS_YES=true /opt/conda/bin/conda env create -n ${PROJ_NAME} -q --file /tmp/conda/env.yaml && \
    sed -i "s/conda activate base/conda activate ${PROJ_NAME}/g" ~/.bashrc && \
    conda clean -afy && \
    rm -rf /tmp/conda

ENV MORPHEUS_SUPPORT_DOCA=ON

COPY ./.devcontainer/docker/optional_deps/doca.sh /tmp/doca/

RUN apt update && \
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC \
    apt install --no-install-recommends -y \
        build-essential \
        libtool \
        automake && \
    apt clean && \
    PKG_ARCH=${TARGETARCH} /tmp/doca/doca.sh /tmp/doca && \
    rm -rf /tmp/doca

# ============ test ==================
FROM base as test

# Add any test only dependencies here.

ARG REAL_ARCH
ARG CUDA_SHORT_VER
ARG PROJ_NAME
ARG PYTHON_VER

RUN apt update && \
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC \
    apt install --no-install-recommends -y \
        openjdk-11-jre-headless && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

RUN rapids-dependency-file-generator \
      --config /tmp/conda/dependencies.yaml \
      --output conda \
      --file-key test \
      --matrix "cuda=${CUDA_SHORT_VER};arch=${REAL_ARCH};py=${PYTHON_VER}" > /tmp/conda/env.yaml && \
    CONDA_ALWAYS_YES=true /opt/conda/bin/conda env create -n ${PROJ_NAME} -q --file /tmp/conda/env.yaml && \
    sed -i "s/conda activate base/conda activate ${PROJ_NAME}/g" ~/.bashrc && \
    conda clean -afy && \
    rm -rf /tmp/conda

COPY ./ci/scripts/download_kafka.py /tmp/scripts/download_kafka.py

# Install camouflage needed for unittests to mock a triton server
RUN source activate ${PROJ_NAME} && \
    npm install -g camouflage-server@0.15 && \
    npm cache clean --force

# Install Kafka
RUN source activate ${PROJ_NAME} && \
    python /tmp/scripts/download_kafka.py && \
    rm -rf /tmp/scripts
