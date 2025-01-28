#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import typing
from pathlib import Path

import pytest

from morpheus_llm.service.vdb.faiss_vdb_service import FaissVectorDBResourceService
from morpheus_llm.service.vdb.faiss_vdb_service import FaissVectorDBService

if (typing.TYPE_CHECKING):
    from langchain_core.embeddings import Embeddings
else:
    lc_core_embeddings = pytest.importorskip("langchain_core.embeddings", reason="langchain_core not installed")
    Embeddings = lc_core_embeddings.Embeddings


class FakeEmbedder(Embeddings):

    def embed_query(self, text: str) -> list[float]:
        # One-hot encoding using length of text
        vec = [float(0.0)] * 1024

        vec[len(text) % 1024] = 1.0

        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)


@pytest.fixture(scope="function", name="faiss_simple_store_dir")
def faiss_simple_store_dir_fixture(tmp_path: Path):

    from langchain_community.vectorstores.faiss import FAISS

    embeddings = FakeEmbedder()

    # create FAISS docstore for testing
    index_store = FAISS.from_texts([str(x) * x for x in range(3)], embeddings, ids=[chr(x + 97) for x in range(3)])

    index_store.save_local(str(tmp_path), index_name="index")

    # create a second index for testing
    other_store = FAISS.from_texts([str(x) * x for x in range(3, 8)],
                                   embeddings,
                                   ids=[chr(x + 97) for x in range(3, 8)])
    other_store.save_local(str(tmp_path), index_name="other_index")

    return str(tmp_path)


@pytest.fixture(scope="function", name="faiss_service")
def faiss_service_fixture(faiss_simple_store_dir: str):
    # Fixture for FAISS service; can edit FAISS docstore instantiated outside fixture if need to change
    #  embedding model, et.
    service = FaissVectorDBService(local_dir=faiss_simple_store_dir, embeddings=FakeEmbedder())
    yield service


def test_load_resource(faiss_service: FaissVectorDBService):

    # Check the default implementation
    resource = faiss_service.load_resource()
    assert isinstance(resource, FaissVectorDBResourceService)

    # Check specifying a name
    resource = faiss_service.load_resource("index")
    assert resource.describe()["index_name"] == "index"

    # Check another name
    resource = faiss_service.load_resource("other_index")
    assert resource.describe()["index_name"] == "other_index"


def test_describe(faiss_service: FaissVectorDBService):
    desc_dict = faiss_service.load_resource().describe()

    assert desc_dict["index_name"] == "index"
    assert os.path.exists(desc_dict["folder_path"])
    # Room for other properties


def test_count(faiss_service: FaissVectorDBService):

    count = faiss_service.load_resource().count()
    assert count == 3


async def test_similarity_search(faiss_service: FaissVectorDBService):

    vdb = faiss_service.load_resource()

    query_vec = await faiss_service.embeddings.aembed_query("22")

    k_1 = await vdb.similarity_search(embeddings=[query_vec], k=1)

    assert len(k_1[0]) == 1
    assert k_1[0][0]["page_content"] == "22"

    k_3 = await vdb.similarity_search(embeddings=[query_vec], k=3)

    assert len(k_3[0]) == 3
    assert k_3[0][0]["page_content"] == "22"

    # Exceed the number of documents in the docstore
    k_5 = await vdb.similarity_search(embeddings=[query_vec], k=vdb.count() + 2)

    assert len(k_5[0]) == vdb.count()
    assert k_5[0][0]["page_content"] == "22"


def test_has_store_object(faiss_service: FaissVectorDBService):
    assert faiss_service.has_store_object("index")

    assert faiss_service.has_store_object("other_index")

    assert not faiss_service.has_store_object("not_an_index")
