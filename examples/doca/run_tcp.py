# Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

import logging

import click

from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.stages.doca.doca_convert_stage import DocaConvertStage
from morpheus.stages.doca.doca_source_stage import DocaSourceStage
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.inference.triton_inference_stage import TritonInferenceStage
from morpheus.stages.output.write_to_file_stage import WriteToFileStage
from morpheus.stages.postprocess.add_classifications_stage import AddClassificationsStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.stages.preprocess.preprocess_nlp_stage import PreprocessNLPStage
from morpheus.utils.logger import configure_logging


@click.command()
@click.option(
    "--pipeline_batch_size",
    default=1024,
    type=click.IntRange(min=1),
    help=("Internal batch size for the pipeline. Can be much larger than the model batch size. "
          "Also used for Kafka consumers"),
)
@click.option(
    "--model_max_batch_size",
    default=32,
    type=click.IntRange(min=1),
    help="Max batch size to use for the model",
)
@click.option(
    "--model_fea_length",
    default=256,
    type=click.IntRange(min=1),
    help="Features length to use for the model",
)
@click.option(
    "--out_file",
    default="doca_output.csv",
    help="File in which to store output",
)
@click.option(
    "--nic_addr",
    help="NIC PCI Address",
    required=True,
)
@click.option(
    "--gpu_addr",
    help="GPU PCI Address",
    required=True,
)
def run_pipeline(pipeline_batch_size, model_max_batch_size, model_fea_length, out_file, nic_addr, gpu_addr):
    # Enable the default logger
    configure_logging(log_level=logging.DEBUG)

    config = Config()
    config.mode = PipelineModes.NLP

    # Below properties are specified by the command line
    config.num_threads = 5
    config.edge_buffer_size = 1024
    config.pipeline_batch_size = pipeline_batch_size
    config.model_max_batch_size = model_max_batch_size
    config.feature_length = model_fea_length

    config.class_labels = [
        'address',
        'bank_acct',
        'credit_card',
        'email',
        'govt_id',
        'name',
        'password',
        'phone_num',
        'secret_keys',
        'user'
    ]

    pipeline = LinearPipeline(config)

    # add doca source stage
    pipeline.set_source(DocaSourceStage(config, nic_addr, gpu_addr, 'tcp'))
    pipeline.add_stage(DocaConvertStage(config))

    # add deserialize stage
    pipeline.add_stage(DeserializeStage(config))
    pipeline.add_stage(MonitorStage(config, description="Deserialize rate", unit='pkts'))

    hashfile = 'data/bert-base-uncased-hash.txt'

    # add preprocessing stage
    pipeline.add_stage(
        PreprocessNLPStage(config,
                           vocab_hash_file=hashfile,
                           do_lower_case=True,
                           truncation=True,
                           add_special_tokens=False,
                           column='data'))

    pipeline.add_stage(MonitorStage(config, description="Tokenize rate", unit='pkts'))

    # add inference stage
    pipeline.add_stage(
        TritonInferenceStage(config,
                             model_name="sid-minibert-onnx",
                             server_url="localhost:8000",
                             force_convert_inputs=True,
                             use_shared_memory=True))

    pipeline.add_stage(MonitorStage(config, description="Inference rate", unit='pkts'))

    # add class stage
    pipeline.add_stage(AddClassificationsStage(config))
    pipeline.add_stage(MonitorStage(config, description="AddClass rate", unit='pkts'))

    # serialize
    pipeline.add_stage(SerializeStage(config))
    pipeline.add_stage(MonitorStage(config, description="Serialize rate", unit='pkts'))

    # write to file
    pipeline.add_stage(WriteToFileStage(config, filename=out_file, overwrite=True))
    pipeline.add_stage(MonitorStage(config, description="Write to file rate", unit='pkts'))

    # Build the pipeline here to see types in the vizualization
    pipeline.build()

    # Run the pipeline
    pipeline.run()


if __name__ == "__main__":
    run_pipeline()
