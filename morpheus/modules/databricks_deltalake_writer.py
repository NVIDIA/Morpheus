# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
import pandas as pd
from mrc.core import operators as ops

import cudf

from morpheus.utils.module_utils import register_module
from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.utils.module_ids import DATABRICKS_DELTALAKE_WRITER
from morpheus.utils.module_ids import MORPHEUS_MODULE_NAMESPACE

logger = logging.getLogger(__name__)

IMPORT_EXCEPTION = None
IMPORT_ERROR_MESSAGE = "DataBricks writer module requires the databricks-connect package to be installed."

try:
    from databricks.connect import DatabricksSession
    from pyspark.sql import types as sql_types
except ImportError as import_exc:
    IMPORT_EXCEPTION = import_exc

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
    
@register_module(DATABRICKS_DELTALAKE_WRITER, MORPHEUS_MODULE_NAMESPACE)
def databricks_deltalake_writer(builder: mrc.Builder):
    module_config = builder.get_current_module_config()
    """module_config contains all the required configuration parameters, that would otherwise be passed to the stage.
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

    delta_path = module_config.get("DELTA_PATH", None)
    delta_table_write_mode = module_config.get("DELTA_WRITE_MODE", "append")
    databricks_host=module_config.get("DATABRICKS_HOST", None)
    databricks_token=module_config.get("DATABRICKS_TOKEN", None)
    databricks_cluster_id=module_config.get("DATABRICKS_CLUSTER_ID", None)

    spark = DatabricksSession.builder.remote(host=databricks_host,token=databricks_token,cluster_id=databricks_cluster_id).getOrCreate()
    def write_to_deltalake(message: MessageMeta):
        df = message.copy_dataframe()
        if isinstance(df, cudf.DataFrame):
            df = df.to_pandas()
        schema = _extract_schema_from_pandas_dataframe(df)
        spark_df = spark.createDataFrame(df, schema=schema)
        spark_df.write \
            .format('delta') \
            .option("mergeSchema", "true") \
            .mode(delta_table_write_mode) \
            .save(delta_path)
        return message
    
    def node_fn(obs: mrc.Observable, sub: mrc.Subscriber):
        obs.pipe(ops.map(write_to_deltalake), ops.flatten(), ops.filter(lambda x: x is not None)).subscribe(sub)

    node = builder.make_node(DATABRICKS_DELTALAKE_WRITER, mrc.core.operators.build(node_fn))

    builder.register_module_input("input", node)
    builder.register_module_output("output", node)
