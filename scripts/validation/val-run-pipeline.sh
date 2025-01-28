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

# Make sure to load the utils and globals
source ${SCRIPT_DIR}/val-globals.sh
source ${SCRIPT_DIR}/val-utils.sh

# Make sure the .tmp folder exists
if [[ ! -d "${MORPHEUS_ROOT}/.tmp" ]]; then
   mkdir -p "${MORPHEUS_ROOT}/.tmp"
fi

function run_pipeline_sid_minibert(){

   INPUT_FILE=$1
   INFERENCE_STAGE=$2
   OUTPUT_FILE=$3
   VAL_FILE=$4
   VAL_OUTPUT=$5

   morpheus --log_level=DEBUG run --num_threads=$(nproc) --pipeline_batch_size=1024 --model_max_batch_size=32 --use_cpp=${USE_CPP} \
      pipeline-nlp --model_seq_length=256 \
      from-file --filename=${INPUT_FILE} \
      deserialize \
      preprocess --vocab_hash_file=${MORPHEUS_ROOT}/python/morpheus/morpheus/data/bert-base-uncased-hash.txt --truncation=True --do_lower_case=True --add_special_tokens=False \
      ${INFERENCE_STAGE} \
      monitor --description "Inference Rate" --smoothing=0.001 --unit inf \
      add-class --prefix="si_" \
      validate --val_file_name=${VAL_FILE} --results_file_name=${VAL_OUTPUT} --overwrite \
      serialize --exclude "id" --exclude "^_ts_" \
      to-file --filename=${OUTPUT_FILE} --overwrite
}

function run_pipeline_sid_bert(){

   INPUT_FILE=$1
   INFERENCE_STAGE=$2
   OUTPUT_FILE=$3
   VAL_FILE=$4
   VAL_OUTPUT=$5

   morpheus --log_level=DEBUG run --num_threads=$(nproc) --pipeline_batch_size=1024 --model_max_batch_size=32 --use_cpp=${USE_CPP} \
      pipeline-nlp --model_seq_length=256 \
      from-file --filename=${INPUT_FILE} \
      deserialize \
      preprocess --vocab_hash_file=${MORPHEUS_ROOT}/python/morpheus/morpheus/data/bert-base-cased-hash.txt --truncation=True --do_lower_case=False --add_special_tokens=False \
      ${INFERENCE_STAGE} \
      monitor --description "Inference Rate" --smoothing=0.001 --unit inf \
      add-class --prefix="si_" \
      validate --val_file_name=${VAL_FILE} --results_file_name=${VAL_OUTPUT} --overwrite \
      serialize \
      to-file --filename=${OUTPUT_FILE} --overwrite
}

function run_pipeline_abp_nvsmi(){

   INPUT_FILE=$1
   INFERENCE_STAGE=$2
   OUTPUT_FILE=$3
   VAL_FILE=$4
   VAL_OUTPUT=$5

   morpheus --log_level=DEBUG run --num_threads=$(nproc) --pipeline_batch_size=1024 --model_max_batch_size=1024 --use_cpp=${USE_CPP} \
      pipeline-fil --columns_file=${MORPHEUS_ROOT}/python/morpheus/morpheus/data/columns_fil.txt \
      from-file --filename=${INPUT_FILE} \
      deserialize \
      preprocess \
      ${INFERENCE_STAGE} \
      monitor --description "Inference Rate" --smoothing=0.001 --unit inf \
      add-class \
      validate --val_file_name=${VAL_FILE} --results_file_name=${VAL_OUTPUT} --overwrite \
      serialize \
      to-file --filename=${OUTPUT_FILE} --overwrite
}

function run_pipeline_phishing_email(){

   INPUT_FILE=$1
   INFERENCE_STAGE=$2
   OUTPUT_FILE=$3
   VAL_FILE=$4
   VAL_OUTPUT=$5

   morpheus --log_level=DEBUG run --num_threads=$(nproc) --pipeline_batch_size=1024 --model_max_batch_size=32 --use_cpp=${USE_CPP} \
      pipeline-nlp --model_seq_length=128 --labels_file=${MORPHEUS_ROOT}/python/morpheus/morpheus/data/labels_phishing.txt \
      from-file --filename=${INPUT_FILE} \
      deserialize \
      preprocess --vocab_hash_file=${MORPHEUS_ROOT}/python/morpheus/morpheus/data/bert-base-uncased-hash.txt --truncation=True --do_lower_case=True --add_special_tokens=False \
      ${INFERENCE_STAGE} \
      monitor --description "Inference Rate" --smoothing=0.001 --unit inf \
      add-class --label=is_phishing --threshold=0.7 \
      validate --val_file_name=${VAL_FILE} --results_file_name=${VAL_OUTPUT} --overwrite \
      serialize \
      to-file --filename=${OUTPUT_FILE} --overwrite
}
