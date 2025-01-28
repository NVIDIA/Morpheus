# Copyright (c) 2023-2025, NVIDIA CORPORATION.
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

import pymilvus
from langchain_community.embeddings import HuggingFaceEmbeddings

from morpheus_llm.llm.services.llm_service import LLMService
from morpheus_llm.llm.services.nemo_llm_service import NeMoLLMService
from morpheus_llm.llm.services.openai_chat_service import OpenAIChatService
from morpheus_llm.service.vdb.milvus_client import DATA_TYPE_MAP
from morpheus_llm.service.vdb.milvus_vector_db_service import MilvusVectorDBService
from morpheus_llm.service.vdb.utils import VectorDBServiceFactory

logger = logging.getLogger(__name__)


def build_huggingface_embeddings(model_name: str, model_kwargs: dict = None, encode_kwargs: dict = None):
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

    return embeddings


def build_llm_service(model_name: str, llm_service: str, tokens_to_generate: int, **model_kwargs):
    lowered_llm_service = llm_service.lower()

    service: LLMService | None = None

    if (lowered_llm_service == 'nemollm'):
        model_kwargs['tokens_to_generate'] = tokens_to_generate
        service = NeMoLLMService()
    elif (lowered_llm_service == 'openai'):
        model_kwargs['max_tokens'] = tokens_to_generate
        service = OpenAIChatService()
    else:
        raise RuntimeError(f"Unsupported LLM service name: {llm_service}")

    return service.get_client(model_name=model_name, **model_kwargs)


def build_milvus_config(resource_schema_config: dict):
    schema_fields = []
    for field_data in resource_schema_config["schema_conf"]["schema_fields"]:
        field_data["dtype"] = DATA_TYPE_MAP.get(field_data["dtype"])
        field_schema = pymilvus.FieldSchema(**field_data)
        schema_fields.append(field_schema.to_dict())

    resource_schema_config["schema_conf"]["schema_fields"] = schema_fields

    return resource_schema_config


def build_default_milvus_config(embedding_size: int):
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
                pymilvus.FieldSchema(name="title",
                                     dtype=pymilvus.DataType.VARCHAR,
                                     description="The title of the RSS Page",
                                     max_length=65_535).to_dict(),
                pymilvus.FieldSchema(name="link",
                                     dtype=pymilvus.DataType.VARCHAR,
                                     description="The URL of the RSS Page",
                                     max_length=65_535).to_dict(),
                pymilvus.FieldSchema(name="summary",
                                     dtype=pymilvus.DataType.VARCHAR,
                                     description="The summary of the RSS Page",
                                     max_length=65_535).to_dict(),
                pymilvus.FieldSchema(name="page_content",
                                     dtype=pymilvus.DataType.VARCHAR,
                                     description="A chunk of text from the RSS Page",
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


def build_milvus_service(embedding_size: int, uri: str = "http://localhost:19530"):
    default_service = build_default_milvus_config(embedding_size)

    vdb_service: MilvusVectorDBService = VectorDBServiceFactory.create_instance("milvus", uri=uri, **default_service)

    return vdb_service
