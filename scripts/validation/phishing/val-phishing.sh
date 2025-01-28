#!/bin/bash
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

set -e +o pipefail
# set -x
# set -v

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Load the utility scripts
source ${SCRIPT_DIR}/../val-run-pipeline.sh

# Get the model/data from the argument. Must be 'email' or 'url'
PHISHING_TYPE=${PHISHING_TYPE:-$1}

PHISHING_INPUT_FILE=${PHISHING_INPUT_FILE:-"${MORPHEUS_ROOT}/models/datasets/validation-data/phishing-${PHISHING_TYPE}-validation-data.jsonlines"}
PHISHING_TRUTH_FILE=${PHISHING_TRUTH_FILE:-"${MORPHEUS_ROOT}/models/datasets/validation-data/phishing-${PHISHING_TYPE}-validation-data.jsonlines"}

MODEL_FILE=${MODEL_FILE:-"${MORPHEUS_ROOT}/models/phishing-models/phishing-bert-20211006.pt"}
MODEL_DIRECTORY=${MODEL_FILE%/*}
MODEL_FILENAME=$(basename -- "${MODEL_FILE}")
MODEL_EXTENSION="${MODEL_FILENAME##*.}"
MODEL_NAME="${MODEL_FILENAME%.*}"

OUTPUT_FILE_BASE="${MORPHEUS_ROOT}/.tmp/val_${MODEL_NAME}-"


if [[ "${RUN_PYTORCH}" = "1" ]]; then
   OUTPUT_FILE="${OUTPUT_FILE_BASE}pytorch.csv"
   VAL_OUTPUT_FILE="${OUTPUT_FILE_BASE}pytorch-results.json"

   run_pipeline_phishing_${PHISHING_TYPE} \
      "${PHISHING_INPUT_FILE}" \
      "inf-pytorch --model_filename=${MODEL_FILE}" \
      "${OUTPUT_FILE}" \
      "${PHISHING_TRUTH_FILE}" \
      "${VAL_OUTPUT_FILE}"

   # Get the diff
   PYTORCH_ERROR="${b}$(calc_error_val ${VAL_OUTPUT_FILE})"
else
   PYTORCH_ERROR="${y}Skipped"
fi

if [[ "${RUN_TRITON_ONNX}" = "1" ]]; then

   load_triton_model "phishing-bert-onnx"

   OUTPUT_FILE="${OUTPUT_FILE_BASE}triton-onnx.csv"
   VAL_OUTPUT_FILE="${OUTPUT_FILE_BASE}triton-onnx-results.json"

   run_pipeline_phishing_${PHISHING_TYPE} \
      "${PHISHING_INPUT_FILE}" \
      "inf-triton --model_name=phishing-bert-onnx --server_url=${TRITON_URL} --force_convert_inputs=True" \
      "${OUTPUT_FILE}" \
      "${PHISHING_TRUTH_FILE}" \
      "${VAL_OUTPUT_FILE}"

   # Get the diff
   TRITON_ONNX_ERROR="${b}$(calc_error_val ${VAL_OUTPUT_FILE})"
else
   TRITON_ONNX_ERROR="${y}Skipped"
fi

if [[ "${RUN_TRITON_TRT}" = "1" ]]; then
   load_triton_model "phishing-bert-trt"

   OUTPUT_FILE="${OUTPUT_FILE_BASE}triton-trt.csv"
   VAL_OUTPUT_FILE="${OUTPUT_FILE_BASE}triton-trt-results.json"

   run_pipeline_phishing_${PHISHING_TYPE} \
      "${PHISHING_INPUT_FILE}" \
      "inf-triton --model_name=phishing-bert-trt --server_url=${TRITON_URL} --force_convert_inputs=True" \
      "${OUTPUT_FILE}" \
      "${PHISHING_TRUTH_FILE}" \
      "${VAL_OUTPUT_FILE}"

   # Get the diff
   TRITON_TRT_ERROR="${b}$(calc_error_val ${VAL_OUTPUT_FILE})"
else
   TRITON_TRT_ERROR="${y}Skipped"
fi

if [[ "${RUN_TENSORRT}" = "1" ]]; then
   # Generate the TensorRT model
   cd ${MORPHEUS_ROOT}/models/triton-model-repo/sid-${SID_TYPE}-trt/1

   echo "Generating the TensorRT model. This may take a minute..."
   morpheus tools onnx-to-trt --input_model ${MODEL_DIRECTORY}/${MODEL_NAME}.onnx --output_model ./sid-${SID_TYPE}-trt_b1-8_b1-16_b1-32.engine --batches 1 8 --batches 1 16 --batches 1 32 --seq_length 256 --max_workspace_size 16000

   cd ${MORPHEUS_ROOT}

   OUTPUT_FILE="${OUTPUT_FILE_BASE}tensorrt.csv"
   VAL_OUTPUT_FILE="${OUTPUT_FILE_BASE}tensorrt-results.json"

   run_pipeline_phishing_${PHISHING_TYPE} \
      "${PHISHING_INPUT_FILE}" \
      "inf-triton --model_name=sid-${SID_TYPE}-trt --server_url=${TRITON_URL} --force_convert_inputs=True" \
      "${OUTPUT_FILE}" \
      "${PHISHING_TRUTH_FILE}" \
      "${VAL_OUTPUT_FILE}"

   # Get the diff
   TRITON_TRT_ERROR="${b}$(calc_error_val ${VAL_OUTPUT_FILE})"

else
   TENSORRT_ERROR="${y}Skipped"
fi

echo -e "${b}===ERRORS===${x}"
echo -e "PyTorch     :${PYTORCH_ERROR}${x}"
echo -e "Triton(ONNX):${TRITON_ONNX_ERROR}${x}"
echo -e "Triton(TRT) :${TRITON_TRT_ERROR}${x}"
echo -e "TensorRT    :${TENSORRT_ERROR}${x}"

echo -e "${g}Complete!${x}"
