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
import os
import typing

import mrc
import mrc.core.operators as ops
import pandas as pd

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.config import ExecutionMode
from morpheus.messages import MessageMeta
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.preallocator_mixin import PreallocatorMixin
from morpheus.pipeline.single_output_source import SingleOutputSource
from morpheus.pipeline.stage_schema import StageSchema

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from langchain.schema import Document

IMPORT_ERROR_MESSAGE = (
    "ArxivSource requires additional dependencies to be installed. Install them by running the following command: "
    "`conda env update --solver=libmamba -n morpheus"
    "--file conda/environments/all_cuda-125_arch-$(arch).yaml --prune`")


@register_stage("from-arxiv")
class ArxivSource(GpuAndCpuMixin, PreallocatorMixin, SingleOutputSource):
    """
    Source stage that downloads PDFs from arxiv and converts them to dataframes.

    This stage requires several additional dependencies to be installed. Install them by running the following command:
    `conda env update --solver=libmamba -n morpheus "
    "--file conda/environments/all_cuda-125_arch-$(arch).yaml --prune`

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    query : `str`
        Query to use for arxiv search.
    cache_dir : `str`, optional
        Directory to store downloaded PDFs in, any PDFs already in the directory will be skipped.
        This directory, will be created if it does not already exist.
    chunk_size : `int`, optional
        The number of characters to split each PDF into. This is used to split the PDF into multiple chunks each chunk
        will be converted into a row in the resulting dataframe. This value must be larger than `chunk_overlap`.
    chunk_overlap: `int`, optional
        When splitting documents into chunks, this is the number of characters that will overlap from the previus
        chunk.
    max_pages: `int`, optional
        Maximum number of PDF pages to parse.
    """

    def __init__(self,
                 c: Config,
                 query: str,
                 cache_dir: str = "./.cache/arvix_source_cache",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 100,
                 max_pages: int = 10000):

        super().__init__(c)

        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
        except ImportError as exc:
            raise ImportError(IMPORT_ERROR_MESSAGE) from exc

        self._query = query
        self._max_pages = max_pages

        if chunk_size <= chunk_overlap:
            raise ValueError(f"chunk_size must be greater than {chunk_overlap}")

        self._text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                             chunk_overlap=chunk_overlap,
                                                             length_function=len)

        self._total_pdfs = 0
        self._total_pages = 0
        self._total_chunks = 0
        self._cache_dir = cache_dir

        if c.execution_mode == ExecutionMode.GPU:
            import cudf
            self._cudf = cudf

    @property
    def name(self) -> str:
        """Return the name of the stage"""
        return "from-arxiv"

    def supports_cpp_node(self) -> bool:
        """Indicates whether or not this stage supports a C++ node"""
        return False

    def compute_schema(self, schema: StageSchema):
        schema.output_schema.set_type(MessageMeta)

    def _build_source(self, builder: mrc.Builder) -> mrc.SegmentObject:

        download_pages = builder.make_source(self.unique_name + "-download", self._generate_frames)
        process_pages = builder.make_node(self.unique_name + "-process", ops.map(self._process_pages))
        process_pages.launch_options.pe_count = 6

        builder.make_edge(download_pages, process_pages)

        splitting_pages = builder.make_node(self.unique_name + "-split", ops.map(self._splitting_pages))
        # splitting_pages.launch_options.pe_count = 4

        builder.make_edge(process_pages, splitting_pages)

        return splitting_pages

    def _generate_frames(self, subscription: mrc.Subscription):
        os.makedirs(self._cache_dir, exist_ok=True)

        try:
            import arxiv
        except ImportError as exc:
            raise ImportError(IMPORT_ERROR_MESSAGE) from exc

        # Since each result contains at least one page, we know the upper-bound is _max_pages results
        # pylint: disable=c-extension-no-member
        search_results = arxiv.Search(
            query=self._query,
            max_results=self._max_pages,
        )

        for x in search_results.results():
            if self.is_stop_requested() or not subscription.is_subscribed():
                break

            full_path = os.path.join(self._cache_dir, x._get_default_filename())

            if (not os.path.exists(full_path)):
                x.download_pdf(self._cache_dir)
                logger.debug("Downloaded: %s", full_path)
            else:
                logger.debug("Using cached: %s", full_path)

            yield full_path

            self._total_pdfs += 1

        logger.debug("Downloading complete %s pages", self._total_pdfs)

    def _process_pages(self, pdf_path: str):
        try:
            from langchain_community.document_loaders import PyPDFLoader
            from pypdf.errors import PdfStreamError
        except ImportError as exc:
            raise ImportError(IMPORT_ERROR_MESSAGE) from exc

        for _ in range(5):
            try:
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()

                self._total_pages += len(documents)

                logger.debug("Processing %s/%s: %s", len(documents), self._total_pages, pdf_path)
                if self._total_pages > self._max_pages:
                    self.request_stop()

                return documents
            except PdfStreamError:
                logger.error("Failed to load PDF (retrying): %s", pdf_path)
                documents = []

        raise RuntimeError(f"Failed to load PDF: {pdf_path}")

    def _splitting_pages(self, documents: list["Document"]):
        texts = self._text_splitter.split_documents(documents)

        self._total_chunks += len(texts)

        df = pd.json_normalize([x.dict() for x in texts])

        # Rename the columns to remove the metadata prefix
        map_cols = {name: name.removeprefix("metadata.") for name in df.columns if name.startswith("metadata.")}

        df.rename(columns=map_cols, inplace=True)

        if self._config.execution_mode == ExecutionMode.GPU:
            df = self._cudf.from_pandas(df)

        return MessageMeta(df)
