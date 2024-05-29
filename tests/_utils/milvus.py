# SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Utilities for testing Morpheus with Milvus"""

import cudf

from morpheus.service.vdb.milvus_vector_db_service import MilvusVectorDBServiceProvider


def populate_milvus(milvus_server_uri: str,
                    collection_name: str,
                    resource_kwargs: dict,
                    df: cudf.DataFrame,
                    overwrite: bool = False):
    milvus_provider = MilvusVectorDBServiceProvider(uri=milvus_server_uri)
    milvus_service = milvus_provider.create()
    milvus_service.create(collection_name, overwrite=overwrite, **resource_kwargs)
    resource_service = milvus_service.load_resource(name=collection_name)
    resource_service.insert_dataframe(name=collection_name, df=df, **resource_kwargs)
    milvus_service.close()
