# Copyright (c) 2024, NVIDIA CORPORATION.
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

from pydantic import BaseModel
from pydantic import Field
from pydantic import validator

from morpheus.utils.module_ids import MORPHEUS_MODULE_NAMESPACE
from morpheus.utils.module_ids import WRITE_TO_VECTOR_DB
from morpheus.utils.module_utils import ModuleLoaderFactory

logger = logging.getLogger(__name__)

WriteToVectorDBLoaderFactory = ModuleLoaderFactory(WRITE_TO_VECTOR_DB, MORPHEUS_MODULE_NAMESPACE)


class WriteToVDBSchema(BaseModel):
    embedding_column_name: str = "embedding"
    recreate: bool = False
    service_provider: str = Field(default_factory=None)
    default_resource_name: str = Field(default_factory=None)
    resource_schemas: dict = Field(default_factory=dict)
    resource_kwargs: dict = Field(default_factory=dict)
    batch_size: int = 1024
    write_time_interval: float = 1.0

    @validator('service_provider', pre=True)
    def validate_service(cls, to_validate):  # pylint: disable=no-self-argument
        if not to_validate:
            raise ValueError("Service must be an instance of VectorDbServiceProvider")
        return to_validate

    @validator('default_resource_name', pre=True)
    def validate_resource_name(cls, to_validate):  # pylint: disable=no-self-argument
        if not to_validate:
            raise ValueError("Resource name must not be None or Empty.")
        return to_validate

    class Config:
        extra = "forbid"
