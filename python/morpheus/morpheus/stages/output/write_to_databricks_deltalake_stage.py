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
import typing

import mrc
import pandas as pd
from mrc.core import operators as ops

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.pipeline.pass_thru_type_mixin import PassThruTypeMixin
from morpheus.pipeline.single_port_stage import SinglePortStage

logger = logging.getLogger(__name__)

IMPORT_EXCEPTION = None
IMPORT_ERROR_MESSAGE = "DataBricksDeltaLakeSinkStage requires the databricks-connect package to be installed."

try:
    from databricks.connect import DatabricksSession
    from pyspark.sql import types as sql_types
except ImportError as import_exc:
    IMPORT_EXCEPTION = import_exc


@register_stage("to-databricks-deltalake")
class DataBricksDeltaLakeSinkStage(PassThruTypeMixin, SinglePortStage):
    """
    Sink stage used to write messages to a DeltaLake table.

    Parameters
    ----------
    config : morpheus.config.Config
        Pipeline configuration instance.
    delta_path : str, default None
        Path of the delta table where the data need to be written or updated.
    databricks_host : str, default None
        URL of Databricks host to connect to.
    databricks_token : str, default None
        Access token for Databricks cluster.
    databricks_cluster_id : str, default None
        Databricks cluster to be used to query the data as per SQL provided.
    delta_table_write_mode: str, default "append"
        Delta table write mode for storing data.
    """

    def __init__(self,
                 config: Config,
                 delta_path: str = None,
                 databricks_host: str = None,
                 databricks_token: str = None,
                 databricks_cluster_id: str = None,
                 delta_table_write_mode: str = "append"):
        if IMPORT_EXCEPTION is not None:
            raise ImportError(IMPORT_ERROR_MESSAGE) from IMPORT_EXCEPTION

        super().__init__(config)
        self.delta_path = delta_path
        self.delta_table_write_mode = delta_table_write_mode
        self.spark = DatabricksSession.builder.remote(host=databricks_host,
                                                      token=databricks_token,
                                                      cluster_id=databricks_cluster_id).getOrCreate()

    @property
    def name(self) -> str:
        return "to-databricks-deltalake"

    def accepted_types(self) -> typing.Tuple:
        """
        Returns accepted input types for this stage.
        """
        return (MessageMeta, )

    def supports_cpp_node(self) -> bool:
        return False

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:

        def write_to_deltalake(meta: MessageMeta):
            """
            convert cudf to spark dataframe
            """
            df = meta.copy_dataframe()
            if not isinstance(df, pd.DataFrame):
                df = df.to_pandas()

            schema = self._extract_schema_from_pandas_dataframe(df)
            spark_df = self.spark.createDataFrame(df, schema=schema)
            spark_df.write \
                .format('delta') \
                .option("mergeSchema", "true") \
                .mode(self.delta_table_write_mode) \
                .save(self.delta_path)
            return meta

        node = builder.make_node(self.unique_name, ops.map(write_to_deltalake))
        builder.make_edge(input_node, node)

        return node

    @staticmethod
    def _extract_schema_from_pandas_dataframe(df: pd.DataFrame) -> "sql_types.StructType":
        """
        Extract approximate schemas from pandas dataframe
        """
        spark_schema = []
        for col, dtype in df.dtypes.items():
            try:
                if dtype == "bool":
                    spark_dtype = sql_types.StructField(col, sql_types.BooleanType())
                elif dtype == "int64":
                    spark_dtype = sql_types.StructField(col, sql_types.LongType())
                elif dtype == "int32":
                    spark_dtype = sql_types.StructField(col, sql_types.IntegerType())
                elif dtype == "float64":
                    spark_dtype = sql_types.StructField(col, sql_types.DoubleType())
                elif dtype == "float32":
                    spark_dtype = sql_types.StructField(col, sql_types.FloatType())
                elif dtype == "datetime64[ns]":
                    spark_dtype = sql_types.StructField(col, sql_types.TimestampType())
                else:
                    spark_dtype = sql_types.StructField(col, sql_types.StringType())
            except Exception as e:
                logger.error("Encountered error %s while converting columns %s with data type %s", e, col, dtype)
                spark_dtype = sql_types.StructField(col, sql_types.StringType())
            spark_schema.append(spark_dtype)
        return sql_types.StructType(spark_schema)
