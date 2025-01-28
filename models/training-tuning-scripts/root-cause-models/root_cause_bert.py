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
python root_cause_bert.py \
--trainingdata ../../datasets/training-data/root-cause-training-data.csv \
--unseenerrors ../../datasets/training-data/root-cause-unseen-errors.csv
"""

import argparse
import time

import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import cudf

from common.sequence_classifier import SequenceClassifier  # pylint: disable=no-name-in-module
from morpheus.utils.seed import manual_seed


def train(trainingdata, unseenerrors):
    """This function splits the data and appends the new errors
    to test set. After running the training it prints the
    evaluation scores."""

    # setting a random seed

    random_seed = 42

    dflogs = pd.read_csv(trainingdata, header=None, names=['label', 'log'])

    dfnewerror = pd.read_csv(unseenerrors, header=None, names=['label', 'log'])

    (x_train, x_test, y_train, y_test) = train_test_split(dflogs, dflogs.label, random_state=random_seed)

    x_train.reset_index(drop=True, inplace=True)

    y_train.reset_index(drop=True, inplace=True)

    x_test = pd.concat([x_test, dfnewerror])

    y_test = pd.concat([y_test, dfnewerror['label']])

    x_test.reset_index(drop=True, inplace=True)

    y_test.reset_index(drop=True, inplace=True)

    dfnewerror = cudf.DataFrame.from_pandas(dfnewerror)

    dflogs = cudf.DataFrame.from_pandas(dflogs)

    seq_classifier = SequenceClassifier("bert-base-uncased",
                                        hash_file="../../../morpheus/data/bert-base-uncased-hash.txt",
                                        num_labels=2)

    manual_seed(random_seed)
    seq_classifier.train_model(x_train['log'], y_train, batch_size=128, epochs=1, learning_rate=3.6e-04)

    timestr = time.strftime('%Y%m%d-%H%M%S')

    seq_classifier.save_model(timestr)

    print(seq_classifier.evaluate_model(x_test['log'], y_test))

    test_preds = seq_classifier.predict(x_test['log'], batch_size=128)

    tests = test_preds.to_numpy()

    true_labels = x_test['label']

    print(f1_score(true_labels, tests))


def main(args):

    train(args.trainingdata, args.unseenerrors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trainingdata', required=True, help='Labelled data in CSV format')
    parser.add_argument('--unseenerrors',
                        required=True,
                        help="""Labelled data to be added to test set for
                        evaluation after training""")

    main(parser.parse_args())
