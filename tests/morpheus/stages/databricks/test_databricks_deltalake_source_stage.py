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

from unittest.mock import patch

from _utils import assert_results
from _utils.dataset_manager import DatasetManager
from morpheus.config import Config
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.databricks_deltalake_source_stage import DataBricksDeltaLakeSourceStage
from morpheus.stages.output.compare_dataframe_stage import CompareDataFrameStage


def test_databricks_deltalake_source_stage_pipe(config: Config, dataset_cudf: DatasetManager):
    """
    Test the DataBricksDeltaLakeSourceStage against a mock spark session which
    will return spark_df converted into a DataFrame with specific rows per page.
    """

    expected_df = dataset_cudf['filter_probs.csv']
    df = expected_df.to_pandas()
    # pylint: disable=unused-variable
    with patch('morpheus.stages.input.databricks_deltalake_source_stage.DatabricksSession') as mock_db_session:  # NOQA
        databricks_deltalake_source_stage = DataBricksDeltaLakeSourceStage(config,
                                                                           spark_query="",
                                                                           items_per_page=10000,
                                                                           databricks_host="",
                                                                           databricks_token="",
                                                                           databricks_cluster_id="")
        databricks_deltalake_source_stage.spark.sql.return_value.withColumn.return_value.select.return_value.\
            withColumn.return_value.where.return_value.toPandas.return_value.drop.return_value = df
        databricks_deltalake_source_stage.spark.sql.return_value.withColumn.return_value.select.return_value. \
            withColumn.return_value.count.return_value = df.shape[0]
        pipe = LinearPipeline(config)
        pipe.set_source(databricks_deltalake_source_stage)
        comp_stage = pipe.add_stage(CompareDataFrameStage(config, expected_df))
        pipe.run()
        assert_results(comp_stage.get_results())
