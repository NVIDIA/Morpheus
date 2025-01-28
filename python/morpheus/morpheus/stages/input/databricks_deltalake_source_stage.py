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

import mrc

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.config import ExecutionMode
from morpheus.messages.message_meta import MessageMeta
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.preallocator_mixin import PreallocatorMixin
from morpheus.pipeline.single_output_source import SingleOutputSource
from morpheus.pipeline.stage_schema import StageSchema

logger = logging.getLogger(__name__)

IMPORT_EXCEPTION = None
IMPORT_ERROR_MESSAGE = "DatabricksDeltaLakeSourceStage requires the databricks-connect package to be installed."

try:
    from databricks.connect import DatabricksSession
    from pyspark.sql import functions as sf
    from pyspark.sql.window import Window
except ImportError as import_exc:
    IMPORT_EXCEPTION = import_exc


@register_stage("from-databricks-deltalake")
class DataBricksDeltaLakeSourceStage(GpuAndCpuMixin, PreallocatorMixin, SingleOutputSource):
    """
    Source stage used to load messages from a DeltaLake table.

    Parameters
    ----------
    config : morpheus.config.Config
        Pipeline configuration instance.
    spark_query : str, default None
        SQL Query that need to be executed to fetch the results from deltalake table.
    items_per_page: int, default 1000
        Number of rows per iteration/page to be fetched from remote spark cluster.
    databricks_host : str, default None
        URL of Databricks host to connect to.
    databricks_token : str, default None
        Access token for Databricks cluster.
    databricks_cluster_id : str, default None
        Databricks cluster to be used to query the data as per SQL provided.
    """

    def __init__(self,
                 config: Config,
                 spark_query: str = None,
                 items_per_page: int = 1000,
                 databricks_host: str = None,
                 databricks_token: str = None,
                 databricks_cluster_id: str = None):
        if IMPORT_EXCEPTION is not None:
            raise ImportError(IMPORT_ERROR_MESSAGE) from IMPORT_EXCEPTION

        super().__init__(config)
        self.spark_query = spark_query
        self.spark = DatabricksSession.builder.remote(host=databricks_host,
                                                      token=databricks_token,
                                                      cluster_id=databricks_cluster_id).getOrCreate()
        self.items_per_page = items_per_page
        self.offset = 0

        if config.execution_mode == ExecutionMode.GPU:
            import cudf
            self._cudf = cudf

    @property
    def name(self) -> str:
        return "from-databricks-deltalake"

    def supports_cpp_node(self) -> bool:
        return False

    def compute_schema(self, schema: StageSchema):
        schema.output_schema.set_type(MessageMeta)

    def _build_source(self, builder: mrc.Builder) -> mrc.SegmentObject:
        return builder.make_source(self.unique_name, self.source_generator)

    def source_generator(self, subscription: mrc.Subscription):
        try:
            spark_df = self.spark.sql(self.spark_query)
            spark_df = spark_df.withColumn('_id', sf.monotonically_increasing_id())
            window = Window.partitionBy(sf.lit(1)).orderBy("_id")
            spark_df = spark_df.select("*").withColumn("_id", sf.row_number().over(window))
            count = spark_df.count()
            while self.offset <= count and subscription.is_subscribed():
                df = spark_df.where(sf.col('_id').between(self.offset, self.offset + self.items_per_page))
                logger.debug("Reading next iteration data between index: \
                    %s and %s",
                             str(self.offset),
                             str(self.offset + self.items_per_page + 1))
                self.offset += self.items_per_page + 1

                df = df.toPandas().drop(["_id"], axis=1)

                if self._config.execution_mode == ExecutionMode.GPU:
                    df = self._cudf.from_pandas(df)

                yield MessageMeta(df=df)

        except Exception as e:
            logger.error(
                "Error occurred while reading data from \
                        DeltaLake and converting to Dataframe: %s",
                e)
            raise
