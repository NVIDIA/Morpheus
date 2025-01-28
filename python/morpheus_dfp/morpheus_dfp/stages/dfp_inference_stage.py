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
"""Inference stage for DFP."""

import logging
import time
import typing

import mrc
from mlflow.tracking.client import MlflowClient
from mrc.core import operators as ops

from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stage_schema import StageSchema
from morpheus_dfp.utils.model_cache import ModelCache
from morpheus_dfp.utils.model_cache import ModelManager

logger = logging.getLogger(f"morpheus.{__name__}")


class DFPInferenceStage(SinglePortStage):
    """
    This stage performs inference on the input data using the model loaded from MLflow.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    model_name_formatter : str, optional
        Format string to control the name of models stored in MLflow. Currently available field names are: `user_id`
        and `user_md5` which is an md5 hexadecimal digest as returned by `hash.hexdigest`.
    """

    def __init__(self, c: Config, model_name_formatter: str = "dfp-{user_id}"):
        super().__init__(c)

        self._client = MlflowClient()
        self._fallback_user = self._config.ae.fallback_username

        self._model_cache: typing.Dict[str, ModelCache] = {}
        self._model_cache_size_max = 10

        self._cache_timeout_sec = 600

        self._model_manager = ModelManager(model_name_formatter=model_name_formatter)

    @property
    def name(self) -> str:
        """Stage name."""
        return "dfp-inference"

    def supports_cpp_node(self):
        """Whether this stage supports a C++ node."""
        return False

    def accepted_types(self) -> typing.Tuple:
        """Accepted input types."""
        return (ControlMessage, )

    def compute_schema(self, schema: StageSchema):
        schema.output_schema.set_type(ControlMessage)

    def get_model(self, user: str) -> ModelCache:
        """
        Return the model for the given user. If a model doesn't exist for the given user, the model for the generic
        user will be returned.
        """
        return self._model_manager.load_user_model(self._client, user_id=user, fallback_user_ids=[self._fallback_user])

    def on_data(self, message: ControlMessage) -> ControlMessage:
        """Perform inference on the input data."""
        if (not message or message.payload().count == 0):
            return None

        start_time = time.time()

        user_df = message.payload().df.to_pandas()
        user_id = message.get_metadata("user_id")

        try:
            model_cache = self.get_model(user_id)

            if (model_cache is None):
                raise RuntimeError(f"Could not find model for user {user_id}")

            loaded_model = model_cache.load_model()

        except Exception:
            logger.exception("Error trying to get model", exc_info=True)
            return None

        post_model_time = time.time()

        results_df = loaded_model.get_results(user_df, return_abs=True)

        # Create an output message to allow setting meta
        output_message = ControlMessage()
        output_message.payload(message.payload())

        for col in list(results_df.columns):
            output_message.payload().set_data(col, results_df[col])

        output_message.payload().set_data('model_version',
                                          f"{model_cache.reg_model_name}:{model_cache.reg_model_version}")

        if logger.isEnabledFor(logging.DEBUG):
            load_model_duration = (post_model_time - start_time) * 1000.0
            get_anomaly_duration = (time.time() - post_model_time) * 1000.0

            logger.debug("Completed inference for user %s. Model load: %s ms, Model infer: %s ms. Start: %s, End: %s",
                         user_id,
                         load_model_duration,
                         get_anomaly_duration,
                         user_df[self._config.ae.timestamp_column_name].min(),
                         user_df[self._config.ae.timestamp_column_name].max())

        return output_message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data), ops.filter(lambda x: x is not None))
        builder.make_edge(input_node, node)

        # node.launch_options.pe_count = self._config.num_threads

        return node
