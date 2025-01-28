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

# pylint: disable=invalid-name
"""
Example Usage:
python sid-minibert-20230424-script.py \
       --training-data ../../datasets/training-data/sid-sample-training-data.csv \
       --model-dir google/bert_uncased_L-4_H-256_A-4 \
       --tokenizer-hash-filepath /resources/bert-base-uncased-hash.txt \
       --output-file sid-minibert-model.pt
"""

import argparse

import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import multilabel_confusion_matrix
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data.dataset import random_split
from torch.utils.dlpack import from_dlpack
from tqdm import trange
from transformers import AdamW
from transformers import AutoModelForSequenceClassification

import cudf
from cudf.core.subword_tokenizer import SubwordTokenizer


def data_preprocessing(training_data):

    # loading csv with header
    df = cudf.read_csv(training_data)

    # column names to use as lables
    label_names = list(df.columns)

    # do not use raw columns as labels
    label_names.remove("data")

    # sorted
    label_names = sorted(label_names)

    # create a dict for mapping id to label name
    label2idx = {t: i for i, t in enumerate(label_names)}
    idx2label = {v: k for k, v in label2idx.items()}

    # convert labels to pytorch tensor
    labels = from_dlpack(df[label_names].to_dlpack()).type(torch.long)

    cased_tokenizer = SubwordTokenizer("resources/bert-base-uncased-hash.txt", do_lower_case=True)

    tokenizer_output = cased_tokenizer(df.data,
                                       max_length=256,
                                       max_num_rows=len(df.data),
                                       padding='max_length',
                                       return_tensors='pt',
                                       truncation=True,
                                       add_special_tokens=True)

    # create dataset
    dataset = TensorDataset(tokenizer_output["input_ids"], tokenizer_output["attention_mask"], labels)

    # use pytorch random_split to create training and validation data subsets
    dataset_size = len(tokenizer_output["input_ids"])
    train_size = int(dataset_size * .8)  # 80/20 split
    training_dataset, validation_dataset = random_split(dataset, (train_size, (dataset_size - train_size)))

    # create dataloaders
    train_dataloader = DataLoader(dataset=training_dataset, shuffle=True, batch_size=32)
    val_dataloader = DataLoader(dataset=validation_dataset, shuffle=False, batch_size=64)
    return train_dataloader, val_dataloader, idx2label


def train_model(model_dir, train_dataloader, idx2label):

    num_labels = len(idx2label)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=num_labels)
    model.train()
    model.cuda()

    # use DataParallel if you have more than one GPU
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # using hyperparameters recommended in orginial BERT paper
    # the optimizer allows us to apply different hyperpameters
    # for specific parameter groups
    # apply weight decay to all parameters other than bias, gamma, and beta
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [{
        'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01
    },
                                    {
                                        'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                                        'weight_decay_rate': 0.0
                                    }]

    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)

    # number of training epochs
    epochs = 1

    # train loop
    for _ in trange(epochs, desc="Epoch"):
        # tracking variables
        tr_loss = 0  # running loss
        nb_tr_examples, nb_tr_steps = 0, 0

        # train the data for one epoch
        for batch in train_dataloader:
            # unpack the inputs from dataloader
            b_input_ids, b_input_mask, b_labels = batch

            # clear out the gradients
            optimizer.zero_grad()

            # forward pass
            outputs = model(b_input_ids, attention_mask=b_input_mask)
            logits = outputs[0]

            # using binary cross-entropy with logits as loss function
            # assigns independent probabilities to each label
            loss_func = BCEWithLogitsLoss()
            # convert labels to float for calculation
            loss = loss_func(logits.view(-1, num_labels), b_labels.type_as(logits).view(-1, num_labels))

            if torch.cuda.device_count() > 1:
                # mean() to average on multi-gpu parallel training
                loss = loss.mean()
            # backward pass
            loss.backward()

            # update parameters and take a step using the computed gradient
            optimizer.step()

            # update tracking variables
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

        print(f"Train loss: {tr_loss / nb_tr_steps}")
    return model


def save_model(model, output_file):

    if torch.cuda.device_count() > 1:
        model = model.module
    torch.save(model, output_file)


def model_eval(model, val_dataloader, idx2label):

    # model to eval mode to evaluate loss on the validation set
    model.eval()

    # variables to gather full output
    logit_preds, true_labels, pred_labels = [], [], []

    # predict
    for batch in val_dataloader:
        # unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            # forward pass
            output = model(b_input_ids, attention_mask=b_input_mask)
            b_logit_pred = output[0]
            b_pred_label = torch.sigmoid(b_logit_pred)
            b_logit_pred = b_logit_pred.detach().cpu().numpy()
            b_pred_label = b_pred_label.detach().cpu().numpy()
            b_labels = b_labels.detach().cpu().numpy()

        logit_preds.extend(b_logit_pred)
        true_labels.extend(b_labels)
        pred_labels.extend(b_pred_label)

    # calculate accuracy, using 0.50 threshold
    threshold = 0.50
    pred_bools = [pl > threshold for pl in pred_labels]
    true_bools = [tl == 1 for tl in true_labels]
    val_f1_accuracy = f1_score(true_bools, pred_bools, average='macro') * 100
    val_flat_accuracy = accuracy_score(true_bools, pred_bools) * 100

    print('F1 Macro Validation Accuracy: ', val_f1_accuracy)
    print('Flat Validation Accuracy: ', val_flat_accuracy)

    for label, cfm in zip(list(idx2label.values()), multilabel_confusion_matrix(true_bools, pred_bools)):
        print(label)
        print(cfm)


def main(args):
    print("Data Preprocessing...")
    train_dataloader, val_dataloader, idx2label = data_preprocessing(args.training_data)
    print("Model Training...")
    model = train_model(args.model_dir, train_dataloader, idx2label)
    save_model(model, args.output_file)
    print("Model Evaluation...")
    model_eval(model, val_dataloader, idx2label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-data",
                        required=True,
                        help="CSV with 'text' and single T/F \
                        field for each label")
    parser.add_argument("--model-dir",
                        required=True,
                        help="Local directory or HuggingFace directory \
                        with model file")
    parser.add_argument("--tokenizer-hash-filepath", required=True, help="hash file for tokenizer vocab")
    parser.add_argument("--output-file", required=True, help="output file to save new model")

    main(parser.parse_args())
