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
Example Usage:
python abp_nvsmi_xgb_training.py \
       --trainingdata \
       ../../datasets/training-data/abp-sample-nvsmi-training-data.json
"""

import argparse

import xgboost as xgb
from sklearn.model_selection import train_test_split

import cudf


def preprocess(trainingdata):

    # read json data

    df = cudf.read_json(trainingdata)

    with open("../../../morpheus/data/columns_fil.txt", "r", encoding='UTF-8') as fh:
        feat_cols = [x.strip() for x in fh.readlines()]

    feat_cols.append("label")
    df = df[feat_cols]

    # print list of columns
    print(feat_cols)

    # print labels
    print(df['label'].unique())

    return df


def train_val_split(df):

    (x_train, x_test, y_train, y_test) = \
        train_test_split(df.drop(['label'],
                         axis=1), df['label'], train_size=0.8,
                         random_state=1)

    return (x_train, x_test, y_train, y_test)


def train(x_train, x_test, y_train, y_test):

    # move to Dmatrix

    dmatrix_train = xgb.DMatrix(x_train, label=y_train)
    dmatrix_validation = xgb.DMatrix(x_test, label=y_test)

    # Set parameters

    params = {
        'tree_method': 'gpu_hist',
        'eval_metric': 'auc',
        'objective': 'binary:logistic',
        'max_depth': 5,
        'learning_rate': 0.1,
    }
    evallist = [(dmatrix_validation, 'validation'), (dmatrix_train, 'train')]
    num_round = 5

    # Train the model

    # pylint: disable=too-many-function-args
    bst = xgb.train(params, dmatrix_train, num_round, evallist)
    return bst


def save_model(model):
    model.save_model('./abp-nvsmi-xgb-20210310.bst')


# def eval(model,X_test, y_test):

#     fm = ForestInference.load(model, output_class=True)
#     fil_preds_gpu = fm.predict(X_test.astype("float32"))

#     y_pred = fil_preds_gpu.to_array()
#     y_true = y_test.to_array()

#     acc = accuracy_score(y_true, y_pred)
#     print("Validation_score: ", acc)


def main(trainingdata):
    print('Preprocessing...')
    (x_train, x_test, y_train, y_test) = \
        train_val_split(preprocess(trainingdata))
    print('Model Training...')
    model = train(x_train, x_test, y_train, y_test)
    print('Saving Model')
    save_model(model)


#     print("Model Evaluation...")
#     eval(model,X_test, y_test)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trainingdata', required=True, help='Labelled data in JSON format')
    args = parser.parse_args()

    main(args.trainingdata)
