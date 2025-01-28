#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pipeline for detecting phishing emails."""

import logging
import os

import click
from recipient_features_stage import RecipientFeaturesStage
from recipient_features_stage_deco import recipient_features_stage

import morpheus
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.pipeline import LinearPipeline
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.inference.triton_inference_stage import TritonInferenceStage
from morpheus.stages.input.file_source_stage import FileSourceStage
from morpheus.stages.output.write_to_file_stage import WriteToFileStage
from morpheus.stages.postprocess.add_scores_stage import AddScoresStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.stages.preprocess.preprocess_nlp_stage import PreprocessNLPStage
from morpheus.utils.logger import configure_logging

MORPHEUS_ROOT = os.environ['MORPHEUS_ROOT']


@click.command()
@click.option("--use_stage_function",
              is_flag=True,
              default=False,
              help="Use the function based version of the recipient features stage instead of the class")
@click.option(
    "--labels_file",
    type=click.Path(exists=True, readable=True),
    default=os.path.join(morpheus.DATA_DIR, 'labels_phishing.txt'),
    help="Specifies a file to read labels from in order to convert class IDs into labels.",
)
@click.option(
    "--vocab_file",
    type=click.Path(exists=True, readable=True),
    default=os.path.join(morpheus.DATA_DIR, 'bert-base-uncased-hash.txt'),
    help="Path to hash file containing vocabulary of words with token-ids.",
)
@click.option(
    "--input_file",
    type=click.Path(exists=True, readable=True),
    default=os.path.join(MORPHEUS_ROOT, 'examples/data/email_with_addresses.jsonlines'),
    help="Input filepath.",
)
@click.option(
    "--model_fea_length",
    default=128,
    type=click.IntRange(min=1),
    help="Features length to use for the model.",
)
@click.option(
    "--model_name",
    default="phishing-bert-onnx",
    help="The name of the model that is deployed on Tritonserver.",
)
@click.option("--server_url", default='localhost:8000', help="Tritonserver url.")
@click.option(
    "--output_file",
    default=".tmp/output/phishing_detections.jsonlines",
    help="The path to the file where the inference output will be saved.",
)
def run_pipeline(use_stage_function: bool,
                 labels_file: str,
                 vocab_file: str,
                 input_file: str,
                 model_fea_length: int,
                 model_name: str,
                 server_url: str,
                 output_file: str):
    """Run the phishing detection pipeline."""
    # Enable the default logger
    configure_logging(log_level=logging.INFO)

    # It's necessary to configure the pipeline for NLP mode
    config = Config()
    config.mode = PipelineModes.NLP

    # Set the thread count to match our cpu count
    config.num_threads = len(os.sched_getaffinity(0))
    config.feature_length = model_fea_length

    with open(labels_file, encoding='UTF-8') as fh:
        config.class_labels = [x.strip() for x in fh]

    # Create a linear pipeline object
    pipeline = LinearPipeline(config)

    # Set source stage
    pipeline.set_source(FileSourceStage(config, filename=input_file, iterative=False))

    # Add our custom stage
    if use_stage_function:
        pipeline.add_stage(recipient_features_stage(config))
    else:
        pipeline.add_stage(RecipientFeaturesStage(config))

    # Add a deserialize stage
    pipeline.add_stage(DeserializeStage(config))

    # Tokenize the input
    pipeline.add_stage(
        PreprocessNLPStage(config,
                           vocab_hash_file=vocab_file,
                           truncation=True,
                           do_lower_case=True,
                           add_special_tokens=False))

    # Add a inference stage
    pipeline.add_stage(
        TritonInferenceStage(
            config,
            model_name=model_name,
            server_url=server_url,
            force_convert_inputs=True,
        ))

    # Monitor the inference rate
    pipeline.add_stage(MonitorStage(config, description="Inference Rate", smoothing=0.001, unit="inf"))

    # Add detection scores
    pipeline.add_stage(AddScoresStage(config, labels=["is_phishing"]))

    # Write the to the output file
    pipeline.add_stage(SerializeStage(config))
    pipeline.add_stage(WriteToFileStage(config, filename=output_file, overwrite=True))

    # Run the pipeline
    pipeline.run()


if __name__ == "__main__":
    run_pipeline()
