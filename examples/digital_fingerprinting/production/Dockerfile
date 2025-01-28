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

ARG BASE_IMG=nvcr.io/nvidia/cuda
ARG BASE_IMG_TAG=12.5.1-base-ubuntu22.04

FROM --platform=$TARGETPLATFORM ${BASE_IMG}:${BASE_IMG_TAG} AS base

# Install necessary dependencies using apt-get
RUN apt-get update && apt-get install -y \
      git \
      git-lfs \
      wget \
    && apt-get clean

# Install miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-$(arch).sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh

# Add conda to the PATH
ENV PATH=/opt/conda/bin:$PATH

SHELL ["/bin/bash", "-c"]

# Set the tracking URI for mlflow
ENV MLFLOW_TRACKING_URI="http://mlflow:5000"

# This will get used by pipelines for the --s3_cache option
# ENV DFP_S3_CACHE="/work/examples/dfp_workflow/morpheus/.s3_cache"

# Set the workdir to be the production folder
WORKDIR /workspace/examples/digital_fingerprinting/production

# If any changes have been made from the base image, recopy the sources
COPY . /workspace/examples/digital_fingerprinting/production/

# Create a conda env with morpheus-dfp and any additional dependencies needed to run the examples
RUN conda env create --solver=libmamba -y --name morpheus-dfp --file ./conda/environments/dfp_example_cuda-125_arch-$(arch).yaml

ENTRYPOINT [ "/opt/conda/envs/morpheus-dfp/bin/tini", "--", "/workspace/examples/digital_fingerprinting/production/docker/entrypoint.sh" ]

SHELL ["/bin/bash", "-c"]

# ===== Setup for running unattended =====
FROM --platform=$TARGETPLATFORM base AS runtime

# Launch morpheus
CMD ["./launch.sh"]

# ===== Setup for running Jupyter =====
FROM --platform=$TARGETPLATFORM base AS jupyter

# Install the jupyter specific requirements
RUN source activate morpheus-dfp &&\
    /opt/conda/bin/conda install --solver=libmamba -y -c conda-forge \
        ipywidgets \
        jupyter_contrib_nbextensions \
        # notebook v7 is incompatible with jupyter_contrib_nbextensions
        notebook=6 &&\
    jupyter contrib nbextension install --user &&\
    pip install jupyterlab_nvdashboard==0.9

# Launch jupyter
CMD ["jupyter-lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
