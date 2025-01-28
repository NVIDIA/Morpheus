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

import mrc
import pandas as pd
from mrc.core import operators as ops

from dask.distributed import Client

import cudf

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline.control_message_stage import ControlMessageStage
from morpheus.pipeline.preallocator_mixin import PreallocatorMixin

from common.data_models import FeatureConfig  # pylint: disable=no-name-in-module # isort: skip
from common.feature_extractor import FeatureExtractor  # pylint: disable=no-name-in-module # isort: skip


@register_stage("create-features", modes=[PipelineModes.FIL])
class CreateFeaturesRWStage(PreallocatorMixin, ControlMessageStage):
    """
    Stage creates features from Appshiled plugins data.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance
    interested_plugins : list[str]
        Only intrested plugins files will be read from Appshield snapshots
    feature_columns : list[str]
        List of features needed to be extracted.
    file_extns : list[str]
        File extensions.
    n_workers: int, default = 2
        Number of dask workers.
    threads_per_worker: int, default = 2
        Number of threads for each dask worker.
    """

    def __init__(
        self,
        c: Config,
        interested_plugins: list[str],
        feature_columns: list[str],
        file_extns: list[str],
        n_workers: int = 2,
        threads_per_worker: int = 2,
    ):
        self._client = Client(threads_per_worker=threads_per_worker, n_workers=n_workers)
        self._feature_config = FeatureConfig(file_extns, interested_plugins)
        self._feas_all_zeros = dict.fromkeys(feature_columns, 0)

        # FeatureExtractor instance to extract features from the snapshots.
        self._fe = FeatureExtractor(self._feature_config)

        super().__init__(c)

    @property
    def name(self) -> str:
        return "create-features-rw"

    def accepted_types(self) -> tuple:
        """
        Returns accepted input types for this stage.
        """
        return (ControlMessage, )

    def supports_cpp_node(self) -> bool:
        return False

    def on_next(self, msg: ControlMessage) -> list[ControlMessage]:

        snapshot_fea_dfs = []

        with msg.payload().mutable_dataframe() as cdf:
            df = cdf.to_pandas()

        msg_source = msg.get_metadata("source")

        # Type cast CommitCharge.
        df["CommitCharge"] = df["CommitCharge"].astype("float").astype("Int32")
        df["Name"] = df["Name"].str.lower()

        # Create PID_Process feature.
        df['PID_Process'] = df.PID + '_' + df.Process

        snapshot_ids = df.snapshot_id.unique()

        if len(snapshot_ids) > 1:
            # Group snapshot rows using snapshot id.
            all_dfs = [df[df.snapshot_id == snapshot_id] for snapshot_id in snapshot_ids]
        else:
            all_dfs = [df]

        extract_func = self._fe.extract_features
        combine_func = FeatureExtractor.combine_features

        # Schedule dask task `extract_features` per snapshot.
        snapshot_fea_dfs = self._client.map(extract_func, all_dfs, feas_all_zeros=self._feas_all_zeros)

        # Combined `extract_features` results.
        features_df = self._client.submit(combine_func, snapshot_fea_dfs)

        # Gather features from all the snapshots.
        features_df = features_df.result()

        # Snapshot sequence will be generated using `source_pid_process`.
        # Determines which source generated the snapshot messages.
        # There's a chance of receiving the same snapshots names from multiple sources(hosts)
        features_df['source_pid_process'] = msg_source + '_' + features_df.pid_process

        # Cast int values to string preventing the df from converting to cuDF.
        features_df['ldrmodules_df_path'] = features_df['ldrmodules_df_path'].astype(str)

        # Sort entries by pid_process and snapshot_id
        features_df = features_df.sort_values(by=["pid_process", "snapshot_id"]).reset_index(drop=True)

        return self.split_messages(msg_source, features_df)

    def split_messages(self, msg_source: str, df: pd.DataFrame) -> list[ControlMessage]:

        output_messages = []

        pid_processes = df.pid_process.unique()

        # Create a unique messaage per pid_process, this assumes the DF has been sorted by the `pid_process` column
        for pid_process in pid_processes:

            pid_process_index = df[df.pid_process == pid_process].index

            start = pid_process_index.min()
            stop = pid_process_index.max() + 1

            cdf = cudf.DataFrame(df.iloc[start:stop])

            out_msg = ControlMessage()
            out_msg.payload(MessageMeta(cdf))
            out_msg.set_metadata("source", msg_source)

            output_messages.append(out_msg)

        return output_messages

    def on_completed(self):
        # Close dask client when pipeline initiates shutdown
        self._client.close()

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name,
                                 ops.map(self.on_next),
                                 ops.on_completed(self.on_completed),
                                 ops.flatten())
        builder.make_edge(input_node, node)

        return node
