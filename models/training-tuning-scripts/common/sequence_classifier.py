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

import logging

import cupy
import torch
from torch.optim import AdamW
from torch.utils.dlpack import to_dlpack
from tqdm import trange
from transformers import AutoModelForSequenceClassification

import cudf
from cudf.core.subword_tokenizer import SubwordTokenizer

from .dataloader import DataLoader
from .dataset import Dataset

log = logging.getLogger(__name__)


class SequenceClassifier:
    """
    Sequence Classifier using BERT. This class provides methods for training/loading BERT models, evaluation and
    prediction.
    """

    def __init__(self, model_or_path: str, hash_file: str, do_lower: bool = True, num_labels: int = 2):

        self._optimizer = None

        self._model = AutoModelForSequenceClassification.from_pretrained(model_or_path, num_labels=num_labels)

        if torch.cuda.is_available():
            self._device = torch.device("cuda")
            self._model = self._model.cuda()
            # self._model = nn.DataParallel(self._model)
        else:
            self._device = torch.device("cpu")

        self._tokenizer = SubwordTokenizer(hash_file, do_lower_case=do_lower)

    def train_model(
        self,
        train_data,
        labels,
        learning_rate=3e-5,
        max_seq_len=128,
        batch_size=32,
        epochs=5,
    ):
        """
        Train the classifier

        :param train_data: text data for training
        :type train_data: cudf.Series
        :param labels: labels for each element in train_data
        :type labels: cudf.Series
        :param learning_rate: learning rate
        :type learning_rate: float
        :param max_seq_len: Limits the length of the sequence returned by tokenizer. If tokenized sentence is shorter
            than max_seq_len, output will be padded with 0s. If the tokenized sentence is longer than max_seq_len it
            will be truncated to max_seq_len.
        :type max_seq_len: int
        :param batch_size: batch size
        :type batch_size: int
        :param epochs: epoch, default is 5
        :type epochs: int

        Examples
        --------
        >>> from cuml.preprocessing.model_selection import train_test_split
        >>> emails_train, emails_test, labels_train, labels_test = train_test_split(train_emails_df,
                                                                                    'label',
                                                                                    train_size=0.8)
        >>> sc.train_model(emails_train, labels_train)
        """
        train_gdf = cudf.DataFrame()
        train_gdf["text"] = train_data
        train_gdf["label"] = labels

        train_dataset = Dataset(train_gdf)
        train_dataloader = DataLoader(train_dataset, batchsize=batch_size)

        self._config_optimizer(learning_rate)
        self._model.train()  # Enable training mode

        for _ in trange(epochs, desc="Epoch"):
            tr_loss = 0  # Tracking variables
            nb_tr_examples, nb_tr_steps = 0, 0
            for df in train_dataloader.get_chunks():
                b_input_ids, b_input_mask = self._bert_uncased_tokenize(df["text"], max_seq_len)

                b_labels = torch.tensor(df["label"].to_numpy()).cuda()
                self._optimizer.zero_grad()  # Clear out the gradients
                loss = self._model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask,
                                   labels=b_labels)[0]  # forwardpass

                loss.sum().backward()
                self._optimizer.step()  # update parameters
                tr_loss += loss.sum().item()  # get a numeric value
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1

            print(f"Train loss: {tr_loss / nb_tr_steps}")

    def evaluate_model(self, test_data, labels, max_seq_len=128, batch_size=32):
        """
        Evaluate trained model

        :param test_data: test data to evaluate model
        :type test_data: cudf.Series
        :param labels: labels for each element in test_data
        :type labels: cudf.Series
        :param max_seq_len: Limits the length of the sequence returned by tokenizer. If tokenized sentence
            is shorter than max_seq_len, output will be padded with 0s. If the tokenized sentence is
            longer than max_seq_len it will be truncated to max_seq_len.
        :type max_seq_len: int
        :param batch_size: batch size
        :type batch_size: int

        Examples
        --------
        >>> from cuml.preprocessing.model_selection import train_test_split
        >>> emails_train, emails_test, labels_train, labels_test = train_test_split(train_emails_df,
                                                                                    'label',
                                                                                    train_size=0.8)
        >>> sc.evaluate_model(emails_test, labels_test)
        """
        self._model.eval()
        test_gdf = cudf.DataFrame()
        test_gdf["text"] = test_data
        test_gdf["label"] = labels

        test_dataset = Dataset(test_gdf)
        test_dataloader = DataLoader(test_dataset, batchsize=batch_size)

        eval_accuracy = 0
        nb_eval_steps = 0
        for df in test_dataloader.get_chunks():
            b_input_ids, b_input_mask = self._bert_uncased_tokenize(df["text"], max_seq_len)
            b_labels = torch.tensor(df["label"].to_numpy())
            with torch.no_grad():
                logits = self._model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[0]

            logits = logits.type(torch.DoubleTensor).to(self._device)
            logits = cupy.fromDlpack(to_dlpack(logits))
            label_ids = b_labels.type(torch.IntTensor).to(self._device)
            label_ids = cupy.fromDlpack(to_dlpack(label_ids))
            temp_eval_accuracy = self._flatten_accuracy(logits, label_ids)

            eval_accuracy += temp_eval_accuracy
            nb_eval_steps += 1

        accuracy = eval_accuracy / nb_eval_steps

        return float(accuracy)

    def predict(self, input_data, max_seq_len=128, batch_size=32):
        """
        Predict the class with the trained model

        :param input_data: input text data for prediction
        :type input_data: cudf.Series
        :param max_seq_len: Limits the length of the sequence returned by tokenizer. If tokenized sentence
            is shorter than max_seq_len, output will be padded with 0s. If the tokenized sentence is
            longer than max_seq_len it will be truncated to max_seq_len.
        :type max_seq_len: int
        :param batch_size: batch size
        :type batch_size: int
        :return: predictions: predictions are labels (0 or 1) based on minimum threshold
        :rtype: cudf.Series

        Examples
        --------
        >>> from cuml.preprocessing.model_selection import train_test_split
        >>> emails_train, emails_test, labels_train, labels_test =
                train_test_split(train_emails_df, 'label', train_size=0.8)
        >>> sc.train_model(emails_train, labels_train)
        >>> predictions = sc.predict(emails_test)
        """

        predict_gdf = cudf.DataFrame()
        predict_gdf["text"] = input_data

        predict_dataset = Dataset(predict_gdf)
        predict_dataloader = DataLoader(predict_dataset, batchsize=batch_size)

        preds = cudf.Series()

        self._model.eval()
        for df in predict_dataloader.get_chunks():
            b_input_ids, b_input_mask = self._bert_uncased_tokenize(df["text"], max_seq_len)
            with torch.no_grad():
                logits = self._model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[0]

            logits = logits.type(torch.DoubleTensor).to(self._device)
            logits = cupy.fromDlpack(to_dlpack(logits))
            b_preds = cupy.argmax(logits, axis=1).flatten()
            b_preds = cudf.Series(b_preds)
            preds = preds.append(b_preds)

        return preds

    def save_model(self, save_to_path="."):
        """
        Save trained model

        :param save_to_path: directory path to save model, default is current directory
        :type save_to_path: str

        Examples
        --------
        >>> from cuml.preprocessing.model_selection import train_test_split
        >>> emails_train, emails_test, labels_train, labels_test = train_test_split(train_emails_df,
                                                                                    'label',
                                                                                    train_size=0.8)
        >>> sc.save_model()
        """

        self._model.save_pretrained(save_to_path)

    def save_checkpoint(self, file_path):
        """
        Save model checkpoint

        :param file_path: file path to save checkpoint
        :type file_path: str

        Examples
        --------
        >>> sc.init_model("bert-base-uncased")  # huggingface pre-trained model
        >>> sc.train_model(train_data, train_labels)
        >>> sc.save_checkpoint(PATH)
        """

        checkpoint = {"state_dict": self._model.module.state_dict()}
        torch.save(checkpoint, file_path)

    def load_checkpoint(self, file_path):
        """
        Load model checkpoint

        :param file_path: file path to load checkpoint
        :type file_path: str

        Examples
        --------
        >>> sc.init_model("bert-base-uncased")  # huggingface pre-trained model
        >>> sc.load_checkpoint(PATH)
        """

        model_dict = torch.load(file_path)
        self._model.module.load_state_dict(model_dict["state_dict"])

    def _config_optimizer(self, learning_rate):
        param_optimizer = list(self._model.named_parameters())
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay_rate": 0.01,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay_rate": 0.0,
            },
        ]
        self._optimizer = AdamW(optimizer_grouped_parameters, learning_rate)

    def _flatten_accuracy(self, preds, labels):
        pred_flat = cupy.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return cupy.sum(pred_flat == labels_flat) / len(labels_flat)

    def _bert_uncased_tokenize(self, strings, max_length):
        """
        converts cudf.Series of strings to two torch tensors- token ids and attention mask with padding
        """
        output = self._tokenizer(strings,
                                 max_length=max_length,
                                 max_num_rows=len(strings),
                                 truncation=True,
                                 add_special_tokens=True,
                                 return_tensors="pt")
        return output['input_ids'].type(torch.long), output['attention_mask'].type(torch.long)
