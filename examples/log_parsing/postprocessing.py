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

import json
import logging
import pathlib
import typing
from collections import defaultdict

import mrc
import pandas as pd
from mrc.core import operators as ops

import cudf

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stage_schema import StageSchema

logger = logging.getLogger(f"morpheus.{__name__}")


@register_stage("log-postprocess", modes=[PipelineModes.NLP])
class LogParsingPostProcessingStage(SinglePortStage):

    def __init__(self, c: Config, vocab_path: pathlib.Path, model_config_path: pathlib.Path):
        """
        Post-processing stage for log parsing pipeline.

        Parameters
        ----------
        c : `morpheus.config.Config`
            The morpheus config
        vocab_path : pathlib.Path, exists = True, dir_okay = False
            Model vocab file to use for post-processing
        model_config_path : pathlib.Path, exists = True, dir_okay = False
            Model config file
        """
        super().__init__(c)

        self._vocab_path = vocab_path
        self._model_config_path = model_config_path

        self._vocab_lookup = {}

        # Explicitly setting the encoding, we know we have unicode chars in this file and we need to avoid issue:
        # https://github.com/nv-morpheus/Morpheus/issues/859
        with open(vocab_path, encoding='UTF-8') as f:
            for index, line in enumerate(f):
                self._vocab_lookup[index] = line.split()[0]

        with open(model_config_path, encoding='UTF-8') as f:
            config = json.load(f)

        self._label_map = {int(k): v for k, v in config["id2label"].items()}

    @property
    def name(self) -> str:
        return "logparsing-postproc"

    def supports_cpp_node(self):
        return False

    def accepted_types(self) -> typing.Tuple:
        return (ControlMessage, )

    def compute_schema(self, schema: StageSchema):
        schema.output_schema.set_type(MessageMeta)

    def _postprocess(self, msg: ControlMessage):
        with msg.payload().mutable_dataframe() as src_df:
            src_index = src_df.index.to_pandas()

        seq_ids = msg.tensors().get_tensor('seq_ids').get()
        infer_pdf = pd.DataFrame({"doc": src_index, "start": seq_ids[:, 1], "stop": seq_ids[:, 2]})

        infer_pdf["confidences"] = msg.tensors().get_tensor('confidences').tolist()
        infer_pdf["labels"] = msg.tensors().get_tensor('labels').tolist()
        infer_pdf["token_ids"] = msg.tensors().get_tensor('input_ids').tolist()

        infer_pdf["confidences"] = infer_pdf.apply(lambda row: row["confidences"][row["start"]:row["stop"]], axis=1)

        infer_pdf["labels"] = infer_pdf.apply(lambda row: row["labels"][row["start"]:row["stop"]], axis=1)

        infer_pdf["token_ids"] = infer_pdf.apply(lambda row: row["token_ids"][row["start"]:row["stop"]], axis=1)

        # aggregated logs
        infer_pdf = infer_pdf.groupby("doc").agg({"token_ids": "sum", "confidences": "sum", "labels": "sum"})

        # parse_by_label
        parsed_dfs = infer_pdf.apply(lambda row: self.__get_label_dicts(row), axis=1, result_type="expand")

        ext_parsed = pd.DataFrame(parsed_dfs[0].tolist())
        parsed_df = pd.DataFrame()
        for label in ext_parsed.columns:
            if label[0] == "B":
                col_name = label[2:]
                if "I-" + col_name in ext_parsed.columns:
                    parsed_df[col_name] = ext_parsed[label] + " " + ext_parsed["I-" + col_name].fillna('')
                else:
                    parsed_df[col_name] = ext_parsed[label]

        # decode cleanup
        parsed_df = self.__decode_cleanup(parsed_df)
        parsed_df["doc"] = parsed_dfs.index
        return MessageMeta(df=cudf.DataFrame.from_pandas(parsed_df))

    def __get_label_dicts(self, row):
        token_dict = defaultdict(str)
        confidence_dict = defaultdict(list)
        new_label = None
        new_confidence = None
        for label, confidence, token_id in zip(row["labels"], row["confidences"], row["token_ids"]):
            text_token = self._vocab_lookup[token_id]
            if text_token[:2] != "##" and text_token[0] != '.':
                # if not a subword use the current label, else use previous
                new_label = label
                new_confidence = confidence

            if new_label is not None and new_confidence is not None:
                if self._label_map[new_label] in token_dict:
                    token_dict[self._label_map[new_label]] = (token_dict[self._label_map[new_label]] + " " + text_token)
                else:
                    token_dict[self._label_map[new_label]] = text_token

                confidence_dict[self._label_map[label]].append(new_confidence)
            else:
                logger.warning("Ignoring unexecpected subword token: %s", text_token)

        return token_dict, confidence_dict

    def __decode_cleanup(self, df):
        df.replace(r"\s+##", "", regex=True, inplace=True)
        df.replace(r"\s+\.+\s", ".", regex=True, inplace=True)
        df.replace(r"\s+:+\s", ":", regex=True, inplace=True)
        df.replace(r"\s+\|+\s", "|", regex=True, inplace=True)
        df.replace(r"\s+\++\s", "+", regex=True, inplace=True)
        df.replace(r"\s+\-+\s", "-", regex=True, inplace=True)
        df.replace(r"\s+\<", "<", regex=True, inplace=True)
        df.replace(r"\<+\s", "<", regex=True, inplace=True)
        df.replace(r"\s+\>", ">", regex=True, inplace=True)
        df.replace(r"\>+\s", ">", regex=True, inplace=True)
        df.replace(r"\s+\=+\s", "=", regex=True, inplace=True)
        df.replace(r"\s+\#+\s", "#", regex=True, inplace=True)
        df.replace(r"\[+\s", "[", regex=True, inplace=True)
        df.replace(r"\s\]", "]", regex=True, inplace=True)
        df.replace(r"\(+\s", "(", regex=True, inplace=True)
        df.replace(r"\s\)", ")", regex=True, inplace=True)
        df.replace(r"\s\"", "\"", regex=True, inplace=True)
        df.replace(r"\"+\s", "\"", regex=True, inplace=True)
        df.replace(r"\\+\s", "\"", regex=True, inplace=True)
        df.replace(r"\s+_+\s", "_", regex=True, inplace=True)
        df.replace(r"\s+/", "/", regex=True, inplace=True)
        df.replace(r"/+\s", "/", regex=True, inplace=True)
        df.replace(r"\s+\?+\s", "?", regex=True, inplace=True)
        df.replace(r"\s+;+\s", "; ", regex=True, inplace=True)

        return df

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:

        # Convert the messages to rows of strings
        node = builder.make_node(self.unique_name, ops.map(self._postprocess))

        builder.make_edge(input_node, node)

        return node
