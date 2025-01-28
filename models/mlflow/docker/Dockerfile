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

ARG FROM_IMAGE="condaforge/miniforge-pypy3"
ARG PYTHON_VER=3.11
ARG TAG=23.11.0-0
FROM ${FROM_IMAGE}:${TAG} AS base

WORKDIR /mlflow

COPY . ./

# Enables "source activate conda"
SHELL ["/bin/bash", "-c"]

# need to do an upgrade in case the base is stale for whatever reason
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC \
    apt update && \
    apt upgrade -y && \
    apt install --no-install-recommends -y \
    git \
    procps && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

RUN source /opt/conda/etc/profile.d/conda.sh && conda deactivate \
    && conda update conda python=${PYTHON_VER} && conda install "tini=0.19" \
    && conda clean --all
RUN /opt/conda/bin/conda env create -f docker/conda/mlflow-env.yml
RUN sed -i 's/conda activate base/conda activate mlflow/g' ~/.bashrc

SHELL ["/opt/conda/bin/conda", "run", "-n", "mlflow", "/bin/bash", "-c"]

ARG TRITON_DIR=/mlflow/triton-inference-server
ARG TRITON_VER=r24.03

RUN mkdir ${TRITON_DIR} && \
    cd ${TRITON_DIR} && \
    git clone -b ${TRITON_VER} --depth 1 -n --filter=tree:0 https://github.com/triton-inference-server/server && \
    source activate mlflow && \
    cd ${TRITON_DIR}/server && \
    git sparse-checkout set --no-cone deploy/mlflow-triton-plugin && \
    cd deploy/mlflow-triton-plugin && \
    pip install .

RUN ln -sf ${TRITON_DIR}/server/deploy/mlflow-triton-plugin/scripts/publish_model_to_mlflow.py /mlflow && \
    mkdir /mlflow/artifacts

EXPOSE 5000

# Set the entrypoint to use the entrypoint.sh script which sets the conda env
ENTRYPOINT [ "/opt/conda/bin/tini", "--", "docker/entrypoint.sh" ]

CMD [ "/bin/bash" ]
