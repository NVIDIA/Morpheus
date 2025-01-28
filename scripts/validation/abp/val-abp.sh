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

ABP_INPUT_FILE=${ABP_INPUT_FILE:-"${MORPHEUS_ROOT}/models/datasets/validation-data/abp-validation-data.jsonlines"}
ABP_TRUTH_FILE=${ABP_TRUTH_FILE:-"${MORPHEUS_ROOT}/models/datasets/validation-data/abp-validation-data.jsonlines"}

# Get the ABP_MODEL from the argument. Must be 'nvsmi' for now
ABP_TYPE=${ABP_TYPE:-$1}

MODEL_FILE=${MODEL_FILE:-"${MORPHEUS_ROOT}/models/abp-models/abp-${ABP_TYPE}-xgb-20210310.bst"}
MODEL_DIRECTORY=${MODEL_FILE%/*}
MODEL_FILENAME=$(basename -- "${MODEL_FILE}")
MODEL_EXTENSION="${MODEL_FILENAME##*.}"
MODEL_NAME="${MODEL_FILENAME%.*}"

OUTPUT_FILE_BASE="${MORPHEUS_ROOT}/.tmp/val_${MODEL_NAME}-"

if [[ "${RUN_PYTORCH}" = "1" ]]; then
   OUTPUT_FILE="${OUTPUT_FILE_BASE}pytorch.csv"
   VAL_OUTPUT_FILE="${OUTPUT_FILE_BASE}pytorch-results.json"

   run_pipeline_nlp_${SID_TYPE} \
      "${PCAP_INPUT_FILE}" \
      "inf-pytorch --model_filename=${MODEL_PYTORCH_FILE}" \
      "${OUTPUT_FILE}" \
      "${ABP_TRUTH_FILE}" \
      "${VAL_OUTPUT_FILE}"

   # Get the diff
   PYTORCH_ERROR="${b}$(calc_error_val ${VAL_OUTPUT_FILE})"
else
   PYTORCH_ERROR="${y}Skipped"
fi

if [[ "${RUN_TRITON_XGB}" = "1" ]]; then

   load_triton_model "abp-${ABP_TYPE}-xgb"

   OUTPUT_FILE="${OUTPUT_FILE_BASE}triton-xgb.csv"
   VAL_OUTPUT_FILE="${OUTPUT_FILE_BASE}triton-onnx-results.json"

   TRITON_IP=${TRITON_IP:-"localhost"}
   run_pipeline_abp_${ABP_TYPE} \
      "${ABP_INPUT_FILE}" \
      "inf-triton --model_name=abp-${ABP_TYPE}-xgb --server_url=${TRITON_URL} --force_convert_inputs=True" \
      "${OUTPUT_FILE}" \
      "${ABP_TRUTH_FILE}" \
      "${VAL_OUTPUT_FILE}"

   # Get the diff
   TRITON_XGB_ERROR="${b}$(calc_error_val ${VAL_OUTPUT_FILE})"
else
   TRITON_XGB_ERROR="${y}Skipped"
fi

if [[ "${RUN_TRITON_TRT}" = "1" ]]; then
   load_triton_model "sid-${SID_TYPE}-trt"

   OUTPUT_FILE="${OUTPUT_FILE_BASE}triton-trt.csv"
   VAL_OUTPUT_FILE="${OUTPUT_FILE_BASE}triton-trt-results.json"

   TRITON_IP=${TRITON_IP:-"localhost"}
   run_pipeline_abp_${SID_TYPE} \
      "${PCAP_INPUT_FILE}" \
      "inf-triton --model_name=sid-${SID_TYPE}-trt --server_url=${TRITON_URL} --force_convert_inputs=True" \
      "${OUTPUT_FILE}" \
      "${ABP_TRUTH_FILE}" \
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

   load_triton_model "sid-${SID_TYPE}-trt"

   OUTPUT_FILE="${OUTPUT_FILE_BASE}tensorrt.csv"
   VAL_OUTPUT_FILE="${OUTPUT_FILE_BASE}tensorrt-results.json"

   TRITON_IP=${TRITON_IP:-"localhost"}
   run_pipeline_abp_${SID_TYPE} \
      "${PCAP_INPUT_FILE}" \
      "inf-triton --model_name=sid-${SID_TYPE}-trt --server_url=${TRITON_URL} --force_convert_inputs=True" \
      "${OUTPUT_FILE}" \
      "${ABP_TRUTH_FILE}" \
      "${VAL_OUTPUT_FILE}"

   # Get the diff
   TRITON_TRT_ERROR="${b}$(calc_error_val ${VAL_OUTPUT_FILE})"

else
   TENSORRT_ERROR="${y}Skipped"
fi

echo -e "${b}===ERRORS===${x}"
echo -e "PyTorch     :${PYTORCH_ERROR}${x}"
echo -e "Triton(XGB) :${TRITON_XGB_ERROR}${x}"
echo -e "Triton(TRT) :${TRITON_TRT_ERROR}${x}"
echo -e "TensorRT    :${TENSORRT_ERROR}${x}"

echo -e "${g}Complete!${x}"
