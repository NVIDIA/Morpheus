# syntax=docker/dockerfile:1.3

# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# ============== LEGEND =================================================== #
#       [conda_bld_morpheus_core] [conda_bld_morpheus]
#               /                  /
# [base] -> [conda_env] -> [base_extended] -> [runtime_conda_create] -> [runtime]
#                               \
#                         [conda_env_dev] -> [development] -> [development_pydbg]
#                               \
#                         [build_docs]
#                               \
#                         [git_clone]
#
#  base: Contains all system packages common across all environments
#  conda_env: Create morpheus conda environment and set it as the new base
#  base_extended: Optional environmental elements, libraries, packages, etc.
#  conda_env_dev: Full development environment with all necessary conda
#    packages.
#  conda_bld_morpheus: Utility stage used to build and cache the Morpheus
#    conda package for runtime environments. (should not be extended)
#  runtime: Full runtime environment with Morpheus installed as a conda
#    package
#  development: Full development environment able to build against a Morpheus
#    source directory.
#  development_pydbg: Development + debug build of cpython and various GDB
#    debugging macros.
#  build_docs: Build the Morpheus documentation
#  git_clone: Clone the Morpheus repository to ensure we do not unintentionally
#    copy local build artifacts
# ========================================================================= #

# Args used in FROM commands must come first
ARG FROM_IMAGE="nvidia/cuda"
ARG CUDA_MAJOR_VER=12
ARG CUDA_MINOR_VER=5
ARG CUDA_REV_VER=1
ARG LINUX_DISTRO=ubuntu
ARG LINUX_VER=22.04

# =====Global Options=====
ARG PYTHON_VER=3.10
# Allows running the docker build from a different directory than MORPHEUS_ROOT. Needed for using Morpheus in a
# submodule
ARG MORPHEUS_ROOT_HOST=.
# Supply a channel alias to use for conda. This is needed if the conda channels go down
ARG CONDA_CHANNEL_ALIAS="https://conda.anaconda.org"
# This should reflect the output of the `arch` command.
ARG REAL_ARCH=x86_64

# ============ Stage: base ============
# Configure the base conda environment
FROM --platform=$TARGETPLATFORM ${FROM_IMAGE}:${CUDA_MAJOR_VER}.${CUDA_MINOR_VER}.${CUDA_REV_VER}-base-${LINUX_DISTRO}${LINUX_VER} AS base

ARG CUDA_MAJOR_VER
ARG CUDA_MINOR_VER
ARG PYTHON_VER
ARG REAL_ARCH

ENV DEBIAN_FRONTEND=noninteractive
ENV REAL_ARCH=${REAL_ARCH}

# Remove the file which breaks apt caching since we will handle it ourselves. See
# https://vsupalov.com/buildkit-cache-mount-dockerfile/
RUN rm -f /etc/apt/apt.conf.d/docker-clean

# Install dependencies to build
RUN --mount=type=cache,id=apt,target=/var/cache/apt \
    apt-get update &&\
    apt-get upgrade -y &&\
    apt-get install --no-install-recommends -y \
        bc \
        build-essential \
        cuda-compiler-${CUDA_MAJOR_VER}-${CUDA_MINOR_VER} \
        cuda-cudart-${CUDA_MAJOR_VER}-${CUDA_MINOR_VER} \
        cuda-cupti-${CUDA_MAJOR_VER}-${CUDA_MINOR_VER} \
        cuda-gdb-${CUDA_MAJOR_VER}-${CUDA_MINOR_VER} \
        curl \
        jq \
        libcublas-${CUDA_MAJOR_VER}-${CUDA_MINOR_VER} \
        libcufft-${CUDA_MAJOR_VER}-${CUDA_MINOR_VER} \
        libcurand-${CUDA_MAJOR_VER}-${CUDA_MINOR_VER} \
        libcusolver-${CUDA_MAJOR_VER}-${CUDA_MINOR_VER} \
        libnuma1 \
        openssh-client \
        pkg-config \
        tar \
        unzip \
        wget \
        zip \
    && rm -rf /var/lib/apt/lists/*

# Enables "source activate conda"
SHELL ["/bin/bash", "-c"]

ENV CUDA_PATH=/usr/local/cuda

# All code will be under /workspace
ENV MORPHEUS_ROOT=/workspace
WORKDIR ${MORPHEUS_ROOT}

COPY --from=condaforge/mambaforge:latest /opt/conda /opt/conda

ENV PATH="${PATH}:/opt/conda/bin"

# Install mamba to speed the solve up
RUN --mount=type=cache,id=conda_pkgs,target=/opt/conda/pkgs,sharing=locked \
    conda config --set ssl_verify false &&\
    conda config --add pkgs_dirs /opt/conda/pkgs &&\
    # Install mamba, boa and git here. Conda build breaks with other git installs
    /opt/conda/bin/mamba install -y -n base -c conda-forge "boa" "git >=2.35.3" "python=${PYTHON_VER}" "tini=0.19"

# ============ Stage: conda_env ============
# Create the conda environment and install all dependencies
FROM --platform=$TARGETPLATFORM base AS conda_env

ARG MORPHEUS_ROOT_HOST

# Create a base environment
RUN --mount=type=cache,id=conda_pkgs,target=/opt/conda/pkgs,sharing=locked \
    # Create the environment and install as little dependencies as possible
    CONDA_ALWAYS_YES=true /opt/conda/bin/mamba create -n morpheus -c conda-forge  \
      python=${PYTHON_VER} && \
    # Activate the environment automatically from .bashrc
    conda init bash && \
    echo "conda activate morpheus" >> ~/.bashrc

# Set the permenant conda channes to use for morpheus
RUN source activate morpheus &&\
    conda config --env --add channels conda-forge &&\
    conda config --env --add channels nvidia &&\
    conda config --env --add channels rapidsai

# Set the entrypoint to use the entrypoint.sh script which sets the conda env
COPY ${MORPHEUS_ROOT_HOST}/docker/entrypoint.sh ./docker/
ENTRYPOINT [ "/opt/conda/bin/tini", "--", "/workspace/docker/entrypoint.sh" ]

# Reset the shell back to normal
SHELL ["/bin/bash", "-c"]

# ============ Stage: conda_bld_morpheus_core ============
# Now build the morpheus conda package
FROM --platform=$TARGETPLATFORM conda_env AS conda_bld_morpheus_core

ARG MORPHEUS_ROOT_HOST
ARG CUDA_MAJOR_VER
ARG CUDA_MINOR_VER

# to upload the image use -
# --build-arg CONDA_TOKEN=<anaconda-token>
# --build-arg CONDA_UPLOAD=upload
ARG CONDA_TOKEN
ARG CONDA_UPLOAD

ENV GIT_CLONE_PROTECTION_ACTIVE=false

RUN --mount=type=cache,id=apt,target=/var/cache/apt \
    apt update && \
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC \
    apt install --no-install-recommends -y \
        cuda-compiler-${CUDA_MAJOR_VER}-${CUDA_MINOR_VER} \
        cuda-cudart-dev-${CUDA_MAJOR_VER}-${CUDA_MINOR_VER} \
        cuda-cupti-dev-${CUDA_MAJOR_VER}-${CUDA_MINOR_VER} \
        cuda-nvml-dev-${CUDA_MAJOR_VER}-${CUDA_MINOR_VER} \
        cuda-nvtx-${CUDA_MAJOR_VER}-${CUDA_MINOR_VER} \
        libcublas-dev-${CUDA_MAJOR_VER}-${CUDA_MINOR_VER} \
        libcufft-dev-${CUDA_MAJOR_VER}-${CUDA_MINOR_VER} \
        libcurand-dev-${CUDA_MAJOR_VER}-${CUDA_MINOR_VER} \
        libcusolver-dev-${CUDA_MAJOR_VER}-${CUDA_MINOR_VER} && \
    rm -rf /var/lib/apt/lists/*

# Copy the source
COPY . ./

RUN --mount=type=cache,id=workspace_cache,target=/workspace/.cache,sharing=locked \
    --mount=type=cache,id=conda_pkgs,target=/opt/conda/pkgs,sharing=locked \
    --mount=type=cache,id=pip_cache,target=/root/.cache/pip,sharing=locked \
    # Install git-lfs before running the build to avoid errors during conda build
    /opt/conda/bin/mamba install -y -n base -c conda-forge "git-lfs" &&\
    source activate base &&\
    git lfs install &&\
    # Need to get around recent versions of git locking paths until they are deemed safe
    git config --global --add safe.directory "*" &&\
    # Change to the morpheus directory and build the conda package
    cd ${MORPHEUS_ROOT_HOST} &&\
    MORPHEUS_PYTHON_BUILD_STUBS=OFF CONDA_TOKEN=${CONDA_TOKEN} CONDA_BLD_PATH=/opt/conda/conda-bld ./ci/conda/recipes/run_conda_build.sh morpheus-core ${CONDA_UPLOAD}

# ============ Stage: base_extended ============
# Base environment with any optional extensions. ex. DOCA libs
FROM --platform=$TARGETPLATFORM conda_env AS base_extended

# Add one or more optional dependencies to the base environment
ARG MORPHEUS_ROOT_HOST
ARG MORPHEUS_SUPPORT_DOCA="FALSE"
# Set this environment variable so it auto builds DOCA
ENV MORPHEUS_SUPPORT_DOCA=${MORPHEUS_SUPPORT_DOCA}

# Copy all of the optional dependency scripts
COPY ${MORPHEUS_ROOT_HOST}/.devcontainer/docker/optional_deps docker/optional_deps

# Install DOCA (If requested)
RUN --mount=type=cache,id=doca,target=/tmp/doca,sharing=locked \
    --mount=type=cache,id=apt,target=/var/cache/apt \
    ./docker/optional_deps/doca.sh /tmp/doca

# ============ Stage: conda_env_dev ============
# Create the development conda environment
FROM --platform=$TARGETPLATFORM base_extended AS conda_env_dev

ARG MORPHEUS_ROOT_HOST
ARG CUDA_MAJOR_VER
ARG CUDA_MINOR_VER
ARG CONDA_CHANNEL_ALIAS
ARG REAL_ARCH

# Copy the development dependencies file
COPY ${MORPHEUS_ROOT_HOST}/conda/environments/dev_cuda-${CUDA_MAJOR_VER}${CUDA_MINOR_VER}_arch-${REAL_ARCH}.yaml ./conda/environments/dev.yaml

# Update the morpheus environment
RUN --mount=type=cache,id=conda_pkgs,target=/opt/conda/pkgs,sharing=locked \
    --mount=type=cache,id=pip_cache,target=/root/.cache/pip,sharing=locked \
    # Temp add channel_alias to get around conda 404 errors
    conda config --env --set channel_alias ${CONDA_CHANNEL_ALIAS} &&\
    /opt/conda/bin/conda env update --solver=libmamba -n morpheus --file conda/environments/dev.yaml &&\
    # Remove channel_alias to use the normal channel in the container
    conda config --env --remove-key channel_alias

# ============ Stage: conda_bld_morpheus ============
# Now build the morpheus conda package
FROM --platform=$TARGETPLATFORM base_extended AS conda_bld_morpheus

ARG MORPHEUS_ROOT_HOST
ARG CUDA_MAJOR_VER
ARG CUDA_MINOR_VER

ENV GIT_CLONE_PROTECTION_ACTIVE=false

RUN --mount=type=cache,id=apt,target=/var/cache/apt \
    apt update && \
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC \
    apt install --no-install-recommends -y \
        cuda-compiler-${CUDA_MAJOR_VER}-${CUDA_MINOR_VER} \
        cuda-cudart-dev-${CUDA_MAJOR_VER}-${CUDA_MINOR_VER} \
        cuda-cupti-dev-${CUDA_MAJOR_VER}-${CUDA_MINOR_VER} \
        cuda-nvml-dev-${CUDA_MAJOR_VER}-${CUDA_MINOR_VER} \
        cuda-nvtx-${CUDA_MAJOR_VER}-${CUDA_MINOR_VER} \
        libcublas-dev-${CUDA_MAJOR_VER}-${CUDA_MINOR_VER} \
        libcufft-dev-${CUDA_MAJOR_VER}-${CUDA_MINOR_VER} \
        libcurand-dev-${CUDA_MAJOR_VER}-${CUDA_MINOR_VER} \
        libcusolver-dev-${CUDA_MAJOR_VER}-${CUDA_MINOR_VER} && \
    rm -rf /var/lib/apt/lists/*

# Copy the source
COPY . ./

RUN --mount=type=cache,id=workspace_cache,target=/workspace/.cache,sharing=locked \
    --mount=type=cache,id=conda_pkgs,target=/opt/conda/pkgs,sharing=locked \
    --mount=type=cache,id=pip_cache,target=/root/.cache/pip,sharing=locked \
    # Install git-lfs before running the build to avoid errors during conda build
    /opt/conda/bin/mamba install -y -n base -c conda-forge "git-lfs" &&\
    source activate base &&\
    git lfs install &&\
    # Need to get around recent versions of git locking paths until they are deemed safe
    git config --global --add safe.directory "*" &&\
    # Change to the morpheus directory and build the conda package
    cd ${MORPHEUS_ROOT_HOST} &&\
    MORPHEUS_PYTHON_BUILD_STUBS=OFF CONDA_BLD_PATH=/opt/conda/conda-bld ./ci/conda/recipes/run_conda_build.sh morpheus

# ============ Stage: build_docs ============
# Now build the morpheus docs
FROM --platform=$TARGETPLATFORM conda_env_dev AS build_docs

ARG MORPHEUS_ROOT_HOST

# Copy the source
COPY . ./

RUN --mount=type=cache,id=workspace_cache,target=/workspace/.cache,sharing=locked \
    --mount=type=bind,from=conda_bld_morpheus,source=/opt/conda/conda-bld,target=/opt/conda/conda-bld \
    --mount=type=cache,id=conda_pkgs,target=/opt/conda/pkgs,sharing=locked \
    --mount=type=cache,id=pip_cache,target=/root/.cache/pip,sharing=locked \
    source activate morpheus &&\
    CONDA_ALWAYS_YES=true /opt/conda/bin/mamba install -n morpheus \
        -c local \
        -c conda-forge \
        -c huggingface \
        -c rapidsai \
        -c rapidsai-nightly \
        -c nvidia \
        -c nvidia/label/dev \
        -c pytorch \
        -c defaults \
        morpheus && \
    # Change to the morpheus directory and build the docs
    cd ${MORPHEUS_ROOT_HOST} &&\
    CMAKE_CONFIGURE_EXTRA_ARGS="-DMORPHEUS_BUILD_DOCS=ON -DMORPHEUS_PYTHON_BUILD_STUBS=OFF -DMORPHEUS_CUDA_ARCHITECTURES=RAPIDS"\
         ./scripts/compile.sh --target morpheus_docs

# ============ Stage: runtime_conda_create ============
# Setup conda for the runtime environment
FROM --platform=$TARGETPLATFORM base_extended AS runtime_conda_create
ARG MORPHEUS_ROOT_HOST
ARG CUDA_MAJOR_VER
ARG CUDA_MINOR_VER
ARG REAL_ARCH

COPY "${MORPHEUS_ROOT_HOST}/conda/environments/runtime_cuda-${CUDA_MAJOR_VER}${CUDA_MINOR_VER}_arch-${REAL_ARCH}.yaml" "./conda/environments/"

# Mount Morpheus conda package build in `conda_bld_morpheus`
RUN --mount=type=bind,from=conda_bld_morpheus,source=/opt/conda/conda-bld,target=/opt/conda/conda-bld \
    --mount=type=cache,id=conda_pkgs,target=/opt/conda/pkgs,sharing=locked \
    --mount=type=cache,id=pip_cache,target=/root/.cache/pip,sharing=locked \
    # CVE-2018-20225 for the base pip, not the env one
    # conda will ignore the request to remove pip
    python -m pip uninstall -y pip && \
    source activate morpheus &&\
    # Install morpheus
    CONDA_ALWAYS_YES=true /opt/conda/bin/mamba install -n morpheus \
        -c local \
        -c conda-forge \
        -c huggingface \
        -c rapidsai \
        -c rapidsai-nightly \
        -c nvidia \
        -c nvidia/label/dev \
        -c pytorch \
        -c defaults \
        morpheus && \
    /opt/conda/bin/conda env update --solver=libmamba -n morpheus --file \
        conda/environments/runtime_cuda-${CUDA_MAJOR_VER}${CUDA_MINOR_VER}_arch-${REAL_ARCH}.yaml

# ============ Stage: git_clone ============
# Perform a clone of the git repository this ensures that when we copy files from the source repository, we aren't
# unintentionally including build artifacts or other files that shouldn't be in the final image
FROM --platform=$TARGETPLATFORM conda_env_dev AS git_clone

ARG MORPHEUS_ROOT_HOST

# Source the morpheus env to pick up the git-lfs package
RUN --mount=type=bind,source=.,target=/opt/host_repo \
    source activate morpheus &&\
    # Use a local clone to allow unpushed commits to be included
    git clone file:///opt/host_repo/${MORPHEUS_ROOT_HOST} /tmp/morpheus_repo &&\
    cd /tmp/morpheus_repo &&\
    git lfs install &&\
    ./scripts/fetch_data.py fetch examples

# ============ Stage: runtime ============
# Setup container for runtime environment
FROM --platform=$TARGETPLATFORM runtime_conda_create AS runtime

ARG MORPHEUS_ROOT_HOST

# Only copy specific files/folders over that are necessary for runtime
COPY --from=git_clone "/tmp/morpheus_repo/conda/environments/*.yaml" "./conda/environments/"
COPY --from=git_clone "/tmp/morpheus_repo/docker" "./docker"
COPY --from=build_docs "/workspace/${MORPHEUS_ROOT_HOST}/build/docs/html" "./docs"
COPY --from=git_clone "/tmp/morpheus_repo/examples" "./examples"
COPY --from=git_clone "/tmp/morpheus_repo/scripts" "./scripts"
COPY --from=git_clone "/tmp/morpheus_repo/*.md" "./"
COPY --from=git_clone "/tmp/morpheus_repo/LICENSE" "./"
COPY --from=git_clone "/tmp/morpheus_repo/thirdparty" "./thirdparty"

RUN /opt/conda/bin/conda clean -afy && \
    # Ensure the conda-bld directory is indexed even if empty
    mkdir -p /opt/conda/conda-bld && \
    /opt/conda/bin/conda index /opt/conda/conda-bld

# Use morpheus by default
CMD [ "morpheus" ]

# ============ Stage: development ============
# Install and configure development only packages
FROM --platform=$TARGETPLATFORM conda_env_dev AS development

ARG CUDA_MAJOR_VER
ARG CUDA_MINOR_VER

RUN --mount=type=cache,id=apt,target=/var/cache/apt \
    apt update && \
    TZ=Etc/UTC \
    apt install --no-install-recommends -y \
        cuda-compiler-${CUDA_MAJOR_VER}-${CUDA_MINOR_VER} \
        cuda-cudart-dev-${CUDA_MAJOR_VER}-${CUDA_MINOR_VER} \
        cuda-cupti-dev-${CUDA_MAJOR_VER}-${CUDA_MINOR_VER} \
        cuda-nvtx-${CUDA_MAJOR_VER}-${CUDA_MINOR_VER} \
        libcublas-dev-${CUDA_MAJOR_VER}-${CUDA_MINOR_VER} \
        libcufft-dev-${CUDA_MAJOR_VER}-${CUDA_MINOR_VER} \
        libcurand-dev-${CUDA_MAJOR_VER}-${CUDA_MINOR_VER} \
        libcusolver-dev-${CUDA_MAJOR_VER}-${CUDA_MINOR_VER} \
        openjdk-11-jre-headless && \
    rm -rf /var/lib/apt/lists/*

# Install camouflage needed for unittests to mock a triton server
RUN source activate morpheus && \
    npm install -g camouflage-server@0.15 && \
    npm cache clean --force && \
    conda clean -afy

# Setup git to allow other users to access /workspace. Requires git 2.35.3 or
# greater. See https://marc.info/?l=git&m=164989570902912&w=2. Only enable for
# development
RUN git config --global --add safe.directory "*"

# ============ Stage: python_debug_bld ===========
# Configure and build cpython with debugging symbols
FROM --platform=$TARGETPLATFORM development AS development_pydbg

COPY ${MORPHEUS_ROOT_HOST}/ci/conda/recipes/python-dbg/ ./ci/conda/recipes/python-dbg
COPY ${MORPHEUS_ROOT_HOST}/ci/conda/recipes/run_conda_build.sh ./ci/conda/recipes/run_conda_build.sh
COPY ${MORPHEUS_ROOT_HOST}/ci/conda/recipes/python_dbg_install.sh ./ci/conda/recipes/python_dbg_install.sh

# Temporary until #68 goes in
ARG MORPHEUS_USER="root"

# Build and install debug cpython
RUN source activate morpheus \
    && MORPHEUS_ROOT=/workspace \
       CONDA_BLD_PATH=/opt/conda/conda-bld \
       ./ci/conda/recipes/run_conda_build.sh pydebug \
    && ./ci/conda/recipes/python_dbg_install.sh \
       -s $( ls /opt/conda/conda-bld/src_cache/Python-${PYTHON_VER}*.tar.xz ) \
       -c $( ls /opt/conda/conda-bld/linux-64/python-${PYTHON_VER}*.tar.bz2 ) \
       -i ./ci/conda/recipes/python-dbg/source \
       -g $(eval echo "~${MORPHEUS_USER}")/.gdbinit
