# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
Using a BERT model to parse raw logs into jsons.

Example Usage:
python log_parsing_inference.py \
    --inputdata ../../datasets/validation-data/log-parsing-validation-data-input.csv \
    --modelfile ../../log-parsing-models/log-parsing-20220418.bin \
    --configfile ../../log-parsing-models/log-parsing-config-20220418.json \
    --vocabfile ../../training-tuning-scripts/log-parsing-models/resources/bert-base-cased-vocab.txt \
    --hashfile ../../training-tuning-scripts/log-parsing-models/resources/bert-base-cased-hash.txt \
    --outputfile parsed-output.jsonlines
"""

import argparse
import json
import string
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as f
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from transformers import BertForTokenClassification

import cudf
from cudf.core.subword_tokenizer import SubwordTokenizer


class Cybert:
    """
    Cyber log parsing using BERT.

    This class provides methods for loading models, prediction, and postprocessing.
    """

    def __init__(self, vocabfile, hashfile):
        """Initalize model, labels and tokenizer vocab."""
        self._model = None
        self._label_map = {}
        # resources_dir = "%s/resources" % os.path.dirname(os.path.realpath(__file__))
        vocabpath = vocabfile
        self._vocab_lookup = {}
        with open(vocabpath, encoding="utf-8") as file:
            for index, line in enumerate(file):
                self._vocab_lookup[index] = line.split()[0]
        self._hashpath = hashfile

        self.tokenizer = SubwordTokenizer(self._hashpath, do_lower_case=False)

    def load_model(self, model_filepath, config_filepath):
        """
        Load cybert model.

        :param model_filepath: Filepath of the model (.pth or .bin) to be loaded
        :type model_filepath: str
        :param config_filepath: Config file (.json) to be used
        :type config_filepath: str
        Examples
        --------
        >>> cyparse = Cybert()
        >>> cyparse.load_model('/path/to/model.bin', '/path/to/config.json')
        """

        with open(config_filepath, encoding="utf-8") as file:
            config = json.load(file)
        self._label_map = {int(k): v for k, v in config["id2label"].items()}
        self._model = BertForTokenClassification.from_pretrained(
            model_filepath,
            config=config_filepath,
        )
        self._model.cuda()
        self._model.eval()
        self._model = nn.DataParallel(self._model)

    def preprocess(self, raw_data_col, stride_len=64, max_seq_len=256):
        """
        Preprocess and tokenize data for cybert model inference.

        :param raw_data_col: logs to be processed
        :type raw_data_col: cudf.Series
        :param stride_len: Max stride length for processing, default is 116
        :type stride_len: int
        :param max_seq_len: Max sequence length for processing, default is 128
        :type max_seq_len: int
        Examples
        --------
        >>> import cudf
        >>> cyparse = Cybert()
        >>> cyparse.load_model('/path/to/model.pth', '/path/to/config.json')
        >>> raw_df = cudf.Series(['Log event 1', 'Log event 2'])
        >>> input_ids, attention_masks, meta_data = cyparse.preprocess(raw_df)
        """
        for symbol in string.punctuation:
            raw_data_col = raw_data_col.str.replace(symbol, ' ' + symbol + ' ')

        max_rows_tensor = len(raw_data_col) * 2

        tokenizer = SubwordTokenizer(self._hashpath, do_lower_case=False)
        tokenizer_output = tokenizer(raw_data_col,
                                     max_length=max_seq_len,
                                     stride=stride_len,
                                     truncation=False,
                                     max_num_rows=max_rows_tensor,
                                     add_special_tokens=False,
                                     return_tensors='pt')
        input_ids = tokenizer_output['input_ids'].type(torch.long)
        att_mask = tokenizer_output['attention_mask'].type(torch.long)
        meta_data = tokenizer_output['metadata']
        del tokenizer_output
        return input_ids, att_mask, meta_data

    def inference(self, raw_data_col, batch_size=64):
        """
        Cybert inference and postprocessing on dataset.

        :param raw_data_col: logs to be processed
        :type raw_data_col: cudf.Series
        :param batch_size: Log data is processed in batches using a Pytorch dataloader.
        The batch size parameter refers to the batch size indicated in torch.utils.data.DataLoader.
        :type batch_size: int
        :return: parsed_df
        :rtype: pandas.DataFrame
        :return: confidence_df
        :rtype: pandas.DataFrame
        Examples
        --------
        >>> import cudf
        >>> cyparse = Cybert()
        >>> cyparse.load_model('/path/to/model.pth', '/path/to/config.json')
        >>> raw_data_col = cudf.Series(['Log event 1', 'Log event 2'])
        >>> processed_df, confidence_df = cy.inference(raw_data_col)
        """
        input_ids, attention_masks, meta_data = self.preprocess(raw_data_col)
        dataset = TensorDataset(input_ids, attention_masks)
        dataloader = DataLoader(dataset=dataset, shuffle=False, batch_size=batch_size)
        confidences_list = []
        labels_list = []
        for _, batch in enumerate(dataloader):
            in_ids, att_masks = batch
            with torch.no_grad():
                logits = self._model(in_ids, att_masks)[0]
            logits = f.softmax(logits, dim=2)
            confidences, labels = torch.max(logits, 2)
            confidences_list.extend(confidences.detach().cpu().numpy().tolist())
            labels_list.extend(labels.detach().cpu().numpy().tolist())
        infer_pdf = pd.DataFrame(meta_data).astype(int)
        infer_pdf.columns = ["doc", "start", "stop"]
        infer_pdf["confidences"] = confidences_list
        infer_pdf["labels"] = labels_list
        infer_pdf["token_ids"] = input_ids.detach().cpu().numpy().tolist()

        del dataset
        del dataloader
        del logits
        del confidences
        del labels
        del confidences_list
        del labels_list
        parsed, confidence = self.__postprocess(infer_pdf)
        return parsed, confidence

    def __postprocess(self, infer_pdf):
        # cut overlapping edges
        infer_pdf["confidences"] = infer_pdf.apply(lambda row: row["confidences"][row["start"]:row["stop"]], axis=1)

        infer_pdf["labels"] = infer_pdf.apply(lambda row: row["labels"][row["start"]:row["stop"]], axis=1)

        infer_pdf["token_ids"] = infer_pdf.apply(lambda row: row["token_ids"][row["start"]:row["stop"]], axis=1)

        # aggregated logs
        infer_pdf = infer_pdf.groupby("doc").agg({"token_ids": "sum", "confidences": "sum", "labels": "sum"})

        # parse_by_label
        parsed_dfs = infer_pdf.apply(lambda row: self.__get_label_dicts(row), axis=1, result_type="expand")
        ext_parsed = pd.DataFrame(parsed_dfs[0].tolist())
        ext_confidence = pd.DataFrame(parsed_dfs[1].tolist())
        parsed = pd.DataFrame()
        confidence = pd.DataFrame()
        ext_confidence = ext_confidence.applymap(np.mean)
        for label in ext_parsed.columns:
            if label[0] == "B":
                col_name = label[2:]
                if "I-" + col_name in ext_parsed.columns:
                    parsed[col_name] = ext_parsed[label] + " " + ext_parsed["I-" + col_name].fillna('')
                    confidence[col_name] = (ext_confidence[label] + ext_confidence[label]) / 2
                else:
                    parsed[col_name] = ext_parsed[label]
                    confidence[col_name] = ext_confidence[label]
        del ext_parsed
        del ext_confidence

        # decode cleanup
        parsed = self.__decode_cleanup(parsed)
        return parsed, confidence

    def __get_label_dicts(self, row):
        token_dict = defaultdict(str)
        confidence_dict = defaultdict(list)
        for label, confidence, token_id in zip(row["labels"], row["confidences"], row["token_ids"]):
            text_token = self._vocab_lookup[token_id]
            if text_token[:2] != "##" and text_token[0] != '.':
                # if not a subword use the current label, else use previous
                new_label = label
                new_confidence = confidence
            if self._label_map[new_label] in token_dict:
                token_dict[self._label_map[new_label]] = (token_dict[self._label_map[new_label]] + " " + text_token)
            else:
                token_dict[self._label_map[new_label]] = text_token
            confidence_dict[self._label_map[label]].append(new_confidence)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputdata", required=True, help="raw logs csv format")
    parser.add_argument("--vocabfile", required=True, help="BERT vocabulary file")
    parser.add_argument("--hashfile", required=True, help="BERT hash file")
    parser.add_argument("--modelfile", required=True, help="pretrained model bin")
    parser.add_argument("--configfile", required=True, help="pretrained model config")
    parser.add_argument("--outputfile", required=True, help="output filename jsonlines")
    args = parser.parse_args()
    log_parse = Cybert(args.vocabfile, args.hashfile)
    log_parse.load_model(args.modelfile, args.configfile)
    logs_df = cudf.read_csv(args.inputdata)
    parsed_df, confidence_df = log_parse.inference(logs_df["raw"])
    parsed_df.to_json(args.outputfile, orient='records', lines=True)
