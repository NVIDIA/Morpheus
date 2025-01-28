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
"""Write to Elasticsearch stage."""

import logging
import typing

import mrc
import mrc.core.operators as ops
import pandas as pd
import yaml

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.controllers.elasticsearch_controller import ElasticsearchController
from morpheus.messages import MessageMeta
from morpheus.pipeline.pass_thru_type_mixin import PassThruTypeMixin
from morpheus.pipeline.single_port_stage import SinglePortStage

logger = logging.getLogger(__name__)


@register_stage("to-elasticsearch", ignore_args=["connection_kwargs_update_func"])
class WriteToElasticsearchStage(PassThruTypeMixin, SinglePortStage):
    """

    This class writes the messages as documents to Elasticsearch.

    Parameters
    ----------
    config : `morpheus.config.Config`
        Pipeline configuration instance.
    index : str
        Logical namespace that holds a collection of documents.
    connection_conf_file : str
        YAML configuration file for Elasticsearch connection kwargs settings.
    raise_on_exception : bool, optional, default: False
        Whether to raise exceptions on Elasticsearch errors.
    refresh_period_secs : int, optional, default: 2400
        The refresh period in seconds for client refreshing.
    connection_kwargs_update_func : typing.Callable, optional, default: None
        Custom function to update connection parameters.
    """

    def __init__(self,
                 config: Config,
                 index: str,
                 connection_conf_file: str,
                 raise_on_exception: bool = False,
                 refresh_period_secs: int = 2400,
                 connection_kwargs_update_func: typing.Callable = None):

        super().__init__(config)

        self._index = index

        try:
            with open(connection_conf_file, "r", encoding="utf-8") as file:
                connection_kwargs = yaml.safe_load(file)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"The specified connection configuration file '{connection_conf_file}' does not exist.") from exc
        except Exception as exc:
            raise RuntimeError(f"An error occurred while loading the configuration file: {exc}") from exc

        if connection_kwargs_update_func:
            connection_kwargs = connection_kwargs_update_func(connection_kwargs)

        self._controller = ElasticsearchController(connection_kwargs=connection_kwargs,
                                                   raise_on_exception=raise_on_exception,
                                                   refresh_period_secs=refresh_period_secs)

    @property
    def name(self) -> str:
        """Returns the name of this stage."""
        return "to-elasticsearch"

    def accepted_types(self) -> typing.Tuple:
        """
        Returns accepted input types for this stage.

        Returns
        -------
        typing.Tuple(`morpheus.pipeline.messages.MessageMeta`, )
            Accepted input types.

        """
        return (MessageMeta, )

    def supports_cpp_node(self):
        """Indicates whether this stage supports a C++ node."""
        return False

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:

        def on_data(meta: MessageMeta) -> MessageMeta:

            self._controller.refresh_client()

            df = meta.copy_dataframe()
            if not isinstance(df, pd.DataFrame):
                df = df.to_pandas()
                logger.debug("Converted cudf of size: %s to pandas dataframe.", len(df))

            self._controller.df_to_parallel_bulk_write(index=self._index, df=df)

            return meta

        to_elasticsearch = builder.make_node(self.unique_name,
                                             ops.map(on_data),
                                             ops.on_completed(self._controller.close_client))

        builder.make_edge(input_node, to_elasticsearch)

        return to_elasticsearch
