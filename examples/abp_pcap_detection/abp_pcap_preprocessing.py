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

import typing
from functools import partial

import cupy as cp
import numpy as np

import cudf

from morpheus.cli.register_stage import register_stage
from morpheus.common import TypeId
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.messages import ControlMessage
from morpheus.messages import InferenceMemoryFIL
from morpheus.stages.preprocess.preprocess_base_stage import PreprocessBaseStage


@register_stage("pcap-preprocess", modes=[PipelineModes.FIL])
class AbpPcapPreprocessingStage(PreprocessBaseStage):

    def __init__(self, c: Config):
        """
        Pre-processing of PCAP data for Anomalous Behavior Profiling Detection Pipeline

        Parameters
        ----------
        c : Config
            The morpheus config
        """

        super().__init__(c)

        self._fea_length = c.feature_length
        self.features = [
            "ack",
            "psh",
            "rst",
            "syn",
            "fin",
            "ppm",
            "data_len",
            "bpp",
            "all",
            "ackpush/all",
            "rst/all",
            "syn/all",
            "fin/all",
        ]
        assert self._fea_length == len(
            self.features
        ), f"Number of features in preprocessing {len(self.features)}, does not match configuration {self._fea_length}"

        # columns required to be added to input message meta
        self.req_cols = ["flow_id", "rollup_time"]

        for req_col in self.req_cols:
            self._needed_columns[req_col] = TypeId.STRING

    @property
    def name(self) -> str:
        return "preprocess-anomaly"

    def supports_cpp_node(self):
        return False

    @staticmethod
    def pre_process_batch(msg: ControlMessage, fea_len: int, fea_cols: typing.List[str],
                          req_cols: typing.List[str]) -> ControlMessage:
        meta = msg.payload()
        assert meta is not None, "Payload is None"
        orig_df = meta.get_data()

        # Converts the int flags field into a binary string
        flags = orig_df["flags"].astype("int8")
        flags_bin_series = (flags // 16 % 2).astype('O') + (flags // 8 % 2).astype('O') + (
            flags // 4 % 2).astype('O') + (flags // 2 % 2).astype('O') + (flags % 2).astype('O')

        flag_array = flags_bin_series.str.findall("[0-1]").list.astype("int8")

        # Expand binary string into an array adding [ack, psh, rst, syn, fin] details from the binary flag
        rename_cols_dct = {0: "ack", 1: "psh", 2: "rst", 3: "syn", 4: "fin"}
        df = cudf.DataFrame({rename_cols_dct[i]: flag_array.list.get(i) for i in range(5)}, index=orig_df.index)

        df["flags_bin"] = flags_bin_series
        df["timestamp"] = orig_df["timestamp"].astype("int64")

        def round_time_kernel(timestamp, rollup_time, secs):
            for i, time in enumerate(timestamp):
                x = time % secs
                y = 1 - (x / secs)
                delta = y * secs
                rollup_time[i] = time + delta

        df = df.apply_rows(
            round_time_kernel,
            incols=["timestamp"],
            outcols={"rollup_time": np.int64},
            kwargs={"secs": 60000000},
        )

        df["rollup_time"] = cudf.to_datetime(df["rollup_time"], unit="us").dt.strftime("%Y-%m-%d %H:%M")

        # creating flow_id "src_ip:src_port=dst_ip:dst_port"
        df["flow_id"] = (orig_df["src_ip"] + ":" + orig_df["src_port"].astype("str") + "=" + orig_df["dest_ip"] + ":" +
                         orig_df["dest_port"].astype("str"))
        agg_dict = {
            "ack": "sum",
            "psh": "sum",
            "rst": "sum",
            "syn": "sum",
            "fin": "sum",
            "data_len": "sum",
            "flow_id": "count",
        }

        df["data_len"] = orig_df["data_len"].astype("int16")

        # group by operation
        grouped_df = df.groupby(["rollup_time", "flow_id"]).agg(agg_dict)

        # Assumption: Each flow corresponds to a single packet flow
        # Given that the roll-up is on 1 minute, packets-per-minute(ppm)=number of flows
        grouped_df.rename(columns={"flow_id": "ppm"}, inplace=True)
        grouped_df.reset_index(inplace=True)

        # bpp - Bytes per packet per flow. In the absence of data on number-of-packets
        # and the assumption that the flow and packet are the same, bpp=bytes/packets
        grouped_df["bpp"] = grouped_df["data_len"] / grouped_df["ppm"]
        grouped_df["all"] = (grouped_df["ack"] + grouped_df["psh"] + grouped_df["rst"] + grouped_df["syn"] +
                             grouped_df["fin"])

        # ackpush/all - Number of flows with ACK+PUSH flags to all flows
        grouped_df["ackpush/all"] = (grouped_df["ack"] + grouped_df["psh"]) / grouped_df["all"]

        # rst/all - Number of flows with RST flag to all flows
        # syn/all - Number of flows with SYN flag to all flows
        # fin/all - Number of flows with FIN flag to all flows
        for col in ["rst", "syn", "fin"]:
            dst_col = f"{col}/all"
            grouped_df[dst_col] = grouped_df[col] / grouped_df["all"]

        # Adding index column to retain the order of input messages.
        df["idx"] = df.index

        df = df[["rollup_time", "flow_id", "idx"]]

        # Joining grouped dataframe entries with input dataframe entries to match input messages count.
        merged_df = df.merge(
            grouped_df,
            left_on=["rollup_time", "flow_id"],
            right_on=["rollup_time", "flow_id"],
            how="left",
            suffixes=("_left", ""),
        )

        merged_df = merged_df.sort_values("idx")

        del df, grouped_df

        # Convert the dataframe to cupy the same way cuml does
        # Explicity casting to float32 to match the model's input, and setting row-major as required by Triton
        data = cp.asarray(merged_df[fea_cols].to_cupy(), order='C', dtype=cp.float32)
        count = data.shape[0]

        for col in req_cols:
            meta.set_data(col, merged_df[col])

        del merged_df

        seq_ids = cp.zeros((count, 3), dtype=cp.uint32)
        seq_ids[:, 0] = cp.arange(0, count, dtype=cp.uint32)
        seq_ids[:, 2] = fea_len - 1

        # Create the inference memory. Keep in mind count here could be > than input count
        memory = InferenceMemoryFIL(count=count, input__0=data, seq_ids=seq_ids)

        infer_message = ControlMessage(msg)
        infer_message.payload(meta)
        infer_message.tensors(memory)

        return infer_message

    def _get_preprocess_fn(self) -> typing.Callable[[ControlMessage], ControlMessage]:
        return partial(AbpPcapPreprocessingStage.pre_process_batch,
                       fea_len=self._fea_length,
                       fea_cols=self.features,
                       req_cols=self.req_cols)
