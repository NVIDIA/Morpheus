# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import typing

import cupy as cp
import mlflow
import mrc
from mrc.core import operators as ops

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.messages import ControlMessage
from morpheus.pipeline.pass_thru_type_mixin import PassThruTypeMixin
from morpheus.pipeline.single_port_stage import SinglePortStage

logger = logging.getLogger(__name__)


@register_stage("mlflow-drift", modes=[PipelineModes.FIL, PipelineModes.NLP, PipelineModes.OTHER])
class MLFlowDriftStage(PassThruTypeMixin, SinglePortStage):
    """
    Report model drift statistics to MLflow.

    Caculates model drift over time and reports the information to MLflow.

    Parameters
    ----------
    c : `morpheus.config.Config`
        The global config.
    tracking_uri : str, optional
        The MLflow tracking URI to connect to the tracking backend. If not specified, MLflow will use 'file:///mlruns'
        relative to the current directory, by default None.
    experiment_name : str, optional
        The experiment name to use in MLflow, by default "Morpheus".
    run_id : str, optional
        The MLflow Run ID to report metrics to. If unspecified, Morpheus will attempt to reuse any previously created
        runs that are still active. Otherwise, a new run will be created. By default, runs are left in an active state.
    labels : typing.List[str], optional
        Converts probability indexes into labels for the MLflow UI. If no labels are specified, the probability labels
        are determined by the pipeline mode., by default None.
    batch_size : int, optional
        The batch size to calculate model drift statistics. Allows for increasing or decreasing how much data is
        reported to MLflow. Default is -1 which will use the pipeline batch_size., by default -1.
    force_new_run : bool, optional
        Whether or not to reuse the most recent run ID in MLflow or create a new one each time the pipeline is run, by
        default False.
    """

    def __init__(self,
                 c: Config,
                 tracking_uri: str = None,
                 experiment_name: str = "Morpheus",
                 run_id: str = None,
                 labels: typing.List[str] = None,
                 batch_size: int = -1,
                 force_new_run: bool = False):
        super().__init__(c)

        self._feature_length = c.feature_length
        self._tracking_uri = tracking_uri
        self._experiment_name = experiment_name
        self._run_id = run_id
        self._labels = c.class_labels if labels is None or len(labels) == 0 else labels

        if (batch_size == -1):
            self._batch_size = c.pipeline_batch_size
        else:
            self._batch_size = batch_size

        if (self._batch_size > c.pipeline_batch_size):
            logger.warning(("Warning: MLFlowDriftStage batch_size (%d) is greater than pipeline_batch_size (%d). "
                            "Reducing stage batch_size to pipeline_batch_size"),
                           self._batch_size,
                           c.pipeline_batch_size)
            self._batch_size = c.pipeline_batch_size

        # Set the active run up
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        if (run_id is None and not force_new_run):
            # Get the current experiment
            experiment = mlflow.get_experiment_by_name(experiment_name)

            # Find all active runs
            active_runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id],
                                             order_by=["attribute.start_time"])

            if (len(active_runs) > 0 and "tags.morpheus.type" in active_runs):
                morpheus_runs = active_runs[active_runs["tags.morpheus.type"] == "drift"]

                if (len(morpheus_runs) > 0):
                    run_id = morpheus_runs.head(n=1)["run_id"].iloc[0]

        mlflow.start_run(run_id, run_name="Model Drift", tags={"morpheus.type": "drift"})

    @property
    def name(self) -> str:
        return "mlflow_drift"

    def accepted_types(self) -> typing.Tuple:
        """
        Accepted input types for this stage are returned.

        Returns
        -------
        typing.Tuple[`morpheus.messages.ControlMessage`]
            Accepted input types.

        """
        return (ControlMessage, )

    def supports_cpp_node(self):
        return False

    def _calc_drift(self, msg: ControlMessage):
        probs_tensor = msg.tensors().get_tensor("probs")

        # All probs in a batch will be calculated
        shifted = cp.abs(probs_tensor - 0.5) + 0.5

        # Make sure the labels list is long enough
        for label in range(len(self._labels), shifted.shape[1]):
            self._labels.append(str(label))

        count = msg.tensor_count()

        for i in list(range(0, count, self._batch_size)):
            start = i
            end = min(start + self._batch_size, count)
            mean = cp.mean(shifted[start:end, :], axis=0, keepdims=True)

            # For each column, report the metric
            metrics = {self._labels[y]: mean[0, y].item() for y in range(mean.shape[1])}

            metrics["total"] = cp.mean(mean).item()

            mlflow.log_metrics(metrics)

        return msg

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:

        # Convert the messages to rows of strings
        node = builder.make_node(self.unique_name, ops.map(self._calc_drift))
        builder.make_edge(input_node, node)

        return node
