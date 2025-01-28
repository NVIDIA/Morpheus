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
import typing

import pandas as pd

from morpheus.utils.column_info import DataFrameInputSchema
from morpheus.utils.type_aliases import DataFrameType
from morpheus.utils.type_utils import is_cudf_type

if typing.TYPE_CHECKING:
    import cudf

logger = logging.getLogger(__name__)


@typing.overload
def process_dataframe(
    df_in: pd.DataFrame,
    input_schema: DataFrameInputSchema,
) -> pd.DataFrame:
    ...


@typing.overload
def process_dataframe(
    df_in: "cudf.DataFrame",
    input_schema: DataFrameInputSchema,
) -> "cudf.DataFrame":
    ...


def process_dataframe(
    df_in: DataFrameType,
    input_schema: DataFrameInputSchema,
) -> DataFrameType:
    """
    Applies column transformations to the input dataframe as defined by the `input_schema`.

    If `input_schema` is an instance of `DataFrameInputSchema`, and it has a 'json_preproc' attribute,
    the function will first flatten the JSON columns and concatenate the results with the original DataFrame.

    Parameters
    ----------
    df_in : Union[pd.DataFrame, cudf.DataFrame]
        The input DataFrame to process.
    input_schema : Union[DataFrameInputSchema]
        Defines the transformations to apply to 'df_in'.
        with JSON columns preprocessed if 'json_preproc' attribute is present.

    Returns
    -------
    Union[pd.DataFrame, cudf.DataFrame]
        The processed DataFrame. If 'df_in' was a pd.DataFrame, the return type is also pd.DataFrame,
        otherwise, it is cudf.DataFrame.

    Note
    ----
    Any transformation that needs to be performed should be defined in 'input_schema'.
    If 'df_in' is a pandas DataFrame, it is temporarily converted into a cudf DataFrame for the transformation.
    """

    output_df = pd.DataFrame()

    is_cudf = is_cudf_type(df_in)
    if (is_cudf):
        df_in = df_in.to_pandas()

    # Iterate over the column info
    for ci in input_schema.column_info:
        try:
            output_df[ci.name] = ci._process_column(df_in)
        except Exception:
            logger.exception("Failed to process column '%s'. Dataframe: \n%s", ci.name, df_in, exc_info=True)
            raise

    if (input_schema.preserve_columns is not None):
        # Get the list of remaining columns not already added
        df_in_columns = set(df_in.columns) - set(output_df.columns)

        # Finally, keep any columns that match the preserve filters
        match_columns = [y for y in df_in_columns if input_schema.preserve_columns.match(y)]

        output_df[match_columns] = df_in[match_columns]

    if (is_cudf):
        import cudf
        return cudf.from_pandas(output_df)

    return output_df
