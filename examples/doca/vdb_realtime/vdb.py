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
import pymilvus

from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.stages.doca.doca_convert_stage import DocaConvertStage
from morpheus.stages.doca.doca_source_stage import DocaSourceStage
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.inference.triton_inference_stage import TritonInferenceStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.stages.preprocess.preprocess_nlp_stage import PreprocessNLPStage
from morpheus.utils.logger import configure_logging
from morpheus_llm.stages.output.write_to_vector_db_stage import WriteToVectorDBStage


def build_milvus_service(embedding_size):

    milvus_resource_kwargs = {
        "index_conf": {
            "field_name": "embedding",
            "metric_type": "L2",
            "index_type": "HNSW",
            "params": {
                "M": 8,
                "efConstruction": 64,
            },
        },
        "schema_conf": {
            "enable_dynamic_field": True,
            "schema_fields": [
                pymilvus.FieldSchema(name="id",
                                     dtype=pymilvus.DataType.INT64,
                                     description="Primary key for the collection",
                                     is_primary=True,
                                     auto_id=True).to_dict(),
                pymilvus.FieldSchema(name="src_ip",
                                     dtype=pymilvus.DataType.VARCHAR,
                                     description="The packet source IP address",
                                     max_length=65_535).to_dict(),
                pymilvus.FieldSchema(name="data",
                                     dtype=pymilvus.DataType.VARCHAR,
                                     description="The packet data",
                                     max_length=65_535).to_dict(),
                pymilvus.FieldSchema(name="embedding",
                                     dtype=pymilvus.DataType.FLOAT_VECTOR,
                                     description="Embedding vectors",
                                     dim=embedding_size).to_dict(),
            ],
            "description": "Test collection schema"
        }
    }

    return milvus_resource_kwargs


@click.command()
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
@click.option(
    "--triton_server_url",
    type=str,
    default="localhost:8001",
    show_default=True,
    help="Triton server URL.",
)
@click.option(
    "--embedding_model_name",
    required=True,
    default='all-MiniLM-L6-v2',
    show_default=True,
    help="The name of the model that is deployed on Triton server",
)
@click.option(
    "--vector_db_uri",
    type=str,
    default="http://localhost:19530",
    show_default=True,
    help="URI for connecting to Vector Database server.",
)
@click.option(
    "--vector_db_resource_name",
    type=str,
    default="vdb_doca",
    show_default=True,
    help="The identifier of the resource on which operations are to be performed in the vector database.",
)
def run_pipeline(nic_addr: str,
                 gpu_addr: str,
                 triton_server_url: str,
                 embedding_model_name: str,
                 vector_db_uri: str,
                 vector_db_resource_name: str):
    # Enable the default logger
    configure_logging(log_level=logging.DEBUG)

    config = Config()
    config.mode = PipelineModes.NLP
    config.pipeline_batch_size = 1024
    config.feature_length = 512
    config.edge_buffer_size = 512
    config.num_threads = 20

    pipeline = LinearPipeline(config)

    # add doca source stage
    pipeline.set_source(DocaSourceStage(config, nic_addr, gpu_addr, 'udp'))
    pipeline.add_stage(DocaConvertStage(config))

    pipeline.add_stage(MonitorStage(config, description="DOCA GPUNetIO Source rate", unit='pkts'))

    pipeline.add_stage(DeserializeStage(config))

    pipeline.add_stage(PreprocessNLPStage(config))

    pipeline.add_stage(
        TritonInferenceStage(config,
                             force_convert_inputs=True,
                             model_name=embedding_model_name,
                             server_url=triton_server_url,
                             use_shared_memory=True))
    pipeline.add_stage(MonitorStage(config, description="Embedding rate", unit='pkts'))

    pipeline.add_stage(
        WriteToVectorDBStage(config,
                             resource_name=vector_db_resource_name,
                             batch_size=16896,
                             recreate=True,
                             service="milvus",
                             uri=vector_db_uri,
                             resource_schemas={"vdb_doca": build_milvus_service(384)}))
    pipeline.add_stage(MonitorStage(config, description="Upload rate", unit='docs'))

    # Build the pipeline here to see types in the vizualization
    pipeline.build()

    # Run the pipeline
    pipeline.run()


if __name__ == "__main__":
    run_pipeline()
