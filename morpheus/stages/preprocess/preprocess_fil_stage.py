# Copyright (c) 2021-2024, NVIDIA CORPORATION.
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
from functools import partial

import cupy as cp
import mrc
import numpy as np
import pandas as pd

import cudf

import morpheus._lib.messages as _messages
import morpheus._lib.stages as _stages
from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.messages import ControlMessage
from morpheus.messages import InferenceMemoryFIL
from morpheus.messages import MultiInferenceFILMessage
from morpheus.messages import MultiInferenceMessage
from morpheus.messages import MultiMessage
from morpheus.stages.preprocess.preprocess_base_stage import PreprocessBaseStage

logger = logging.getLogger(__name__)


@register_stage("preprocess", modes=[PipelineModes.FIL])
class PreprocessFILStage(PreprocessBaseStage):
    """
    Prepare FIL input DataFrames for inference.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.

    """

    def __init__(self, c: Config):
        super().__init__(c)

        self._fea_length = c.feature_length
        self.features = c.fil.feature_columns

        assert self._fea_length == len(self.features), \
            f"Number of features in preprocessing {len(self.features)}, does not match configuration {self._fea_length}"

    @property
    def name(self) -> str:
        return "preprocess-fil"

    def supports_cpp_node(self):
        return True

    @staticmethod
    def pre_process_batch(msg: ControlMessage, fea_len: int, fea_cols: typing.List[str]) -> ControlMessage:
        """
        For FIL category usecases, this function performs pre-processing.

        Parameters
        ----------
        msg : `morpheus.messages.ControlMessage`
            Input rows received from Deserialized stage.
        fea_len : int
            Number features are being used in the inference.
        fea_cols : typing.Tuple[str]
            List of columns that are used as features.

        Returns
        -------
        `morpheus.messages.ControlMessage`

        """
        try:
            df: cudf.DataFrame = msg.payload().get_data(fea_cols)
        except KeyError:
            logger.exception("Requested feature columns does not exist in the dataframe.", exc_info=True)
            raise

        # Extract just the numbers from each feature col. Not great to operate on x.meta.df here but the operations will
        # only happen once.
        for col in fea_cols:
            if (df[col].dtype == np.dtype(str) or df[col].dtype == np.dtype(object)):
                # If the column is a string, parse the number
                df[col] = df[col].str.extract(r"(\d+)", expand=False).astype("float32")
            elif (df[col].dtype != np.float32):
                # Convert to float32
                df[col] = df[col].astype("float32")

        if (isinstance(df, pd.DataFrame)):
            df = cudf.from_pandas(df)

        # Convert the dataframe to cupy the same way cuml does
        data = cp.asarray(df.to_cupy())

        count = data.shape[0]

        seg_ids = cp.zeros((count, 3), dtype=cp.uint32)
        seg_ids[:, 0] = cp.arange(0, count, dtype=cp.uint32)
        seg_ids[:, 2] = fea_len - 1

        # We need the C++ impl of TensorMemory until #1646 is resolved
        msg.tensors(_messages.TensorMemory(count=count, tensors={"input__0": data, "seq_ids": seg_ids}))
        # msg.set_metadata("inference_memory_params", {"inference_type": "fil"})
        return msg

    def _get_preprocess_fn(
        self
    ) -> typing.Callable[[ControlMessage],
                         ControlMessage]:
        return partial(PreprocessFILStage.pre_process_batch, fea_len=self._fea_length, fea_cols=self.features)

    def _get_preprocess_node(self, builder: mrc.Builder):
        return _stages.PreprocessFILStage(builder, self.unique_name, self.features)
