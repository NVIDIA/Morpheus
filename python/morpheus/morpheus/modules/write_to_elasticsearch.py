# Copyright (c) 2022-2025, NVIDIA CORPORATION.
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
import pickle

import mrc
from mrc.core import operators as ops

from morpheus.controllers.elasticsearch_controller import ElasticsearchController
from morpheus.messages import ControlMessage
from morpheus.utils.module_ids import MORPHEUS_MODULE_NAMESPACE
from morpheus.utils.module_ids import WRITE_TO_ELASTICSEARCH
from morpheus.utils.module_utils import register_module

logger = logging.getLogger(__name__)


@register_module(WRITE_TO_ELASTICSEARCH, MORPHEUS_MODULE_NAMESPACE)
def write_to_elasticsearch(builder: mrc.Builder):
    """
    This module reads input data stream, converts each row of data to a document format suitable for Elasticsearch,
    and writes the documents to the specified Elasticsearch index using the Elasticsearch bulk API.

    Parameters
    ----------
    builder : mrc.Builder
        An mrc Builder object.
    """

    config = builder.get_current_module_config()

    index = config.get("index", None)

    if index is None:
        raise ValueError("Index must not be None.")

    connection_kwargs = config.get("connection_kwargs")

    if not isinstance(connection_kwargs, dict):
        raise ValueError(f"Expects `connection_kwargs` as a dictionary, but it is of type {type(connection_kwargs)}")

    raise_on_exception = config.get("raise_on_exception", False)
    pickled_func_config = config.get("pickled_func_config", None)
    refresh_period_secs = config.get("refresh_period_secs", 2400)

    if pickled_func_config:
        pickled_func_str = pickled_func_config.get("pickled_func_str")
        encoding = pickled_func_config.get("encoding")

        if pickled_func_str and encoding:
            connection_kwargs_update_func = pickle.loads(bytes(pickled_func_str, encoding))
            connection_kwargs = connection_kwargs_update_func(connection_kwargs)

    controller = ElasticsearchController(connection_kwargs=connection_kwargs,
                                         raise_on_exception=raise_on_exception,
                                         refresh_period_secs=refresh_period_secs)

    def on_data(message: ControlMessage):

        df = message.payload().df.to_pandas()

        controller.df_to_parallel_bulk_write(index=index, df=df)

        return message

    node = builder.make_node(WRITE_TO_ELASTICSEARCH, ops.map(on_data), ops.on_completed(controller.close_client))

    # Register input and output port for a module.
    builder.register_module_input("input", node)
    builder.register_module_output("output", node)
