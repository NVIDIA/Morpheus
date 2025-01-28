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
# pylint: disable=invalid-name
"""
Example Usage:
python hammah-20211017-script.py \
       --trainingdata "../../datasets/training-data/dfp-cloudtrail-user123-training-data.csv" \
       --valdata "../../datasets/validation-data/dfp-cloudtrail-user123-validation-data-input.csv"
"""

import argparse

import dill
import pandas as pd
import torch

from morpheus.models.dfencoder import AutoEncoder
from morpheus.utils.seed import manual_seed


def main(args):
    x_train = pd.read_csv(args.trainingdata)
    x_val = pd.read_csv(args.valdata)

    features = [
        'eventSource',
        'eventName',
        'sourceIPAddress',
        'userAgent',
        'userIdentitytype',
        'requestParametersroleArn',
        'requestParametersroleSessionName',
        'requestParametersdurationSeconds',
        'responseElementsassumedRoleUserassumedRoleId',
        'responseElementsassumedRoleUserarn',
        'apiVersion',
        'userIdentityprincipalId',
        'userIdentityarn',
        'userIdentityaccountId',
        'userIdentityaccessKeyId',
        'userIdentitysessionContextsessionIssuerprincipalId',
        'userIdentitysessionContextsessionIssueruserName',
        'tlsDetailsclientProvidedHostHeader',
        'requestParametersownersSetitems',
        'requestParametersmaxResults',
        'requestParametersinstancesSetitems',
        'errorCode',
        'errorMessage',
        'requestParametersmaxItems',
        'responseElementsrequestId',
        'responseElementsinstancesSetitems',
        'requestParametersgroupSetitems',
        'requestParametersinstanceType',
        'requestParametersmonitoringenabled',
        'requestParametersdisableApiTermination',
        'requestParametersebsOptimized',
        'responseElementsreservationId',
        'requestParametersgroupName'
    ]  # NO userIdentitysessionContextsessionIssuerarn,userIdentityuserName
    for i in list(x_train):
        if i not in features:
            x_train = x_train.drop(i, axis=1)
    for i in list(x_val):
        if i not in features:
            x_val = x_val.drop(i, axis=1)

    x_train = x_train.dropna(axis=1, how='all')
    x_val = x_val.dropna(axis=1, how='all')

    for i in list(x_val):
        if i not in list(x_train):
            x_val = x_val.drop([i], axis=1)

    for i in list(x_train):
        if i not in list(x_val):
            x_train = x_train.drop([i], axis=1)
    manual_seed(42)
    model = AutoEncoder(
        encoder_layers=[512, 500],  # layers of the encoding part
        decoder_layers=[512],  # layers of the decoding part
        activation='relu',  # activation function
        swap_probability=0.2,  # noise parameter
        learning_rate=0.01,  # learning rate
        learning_rate_decay=.99,  # learning decay
        batch_size=512,
        logger='ipynb',
        verbose=False,
        optimizer='sgd',  # SGD optimizer is selected(Stochastic gradient descent)
        scaler='gauss_rank',  # feature scaling method
        min_cats=1  # cut off for minority categories
    )

    model.fit(x_train, epochs=25, validation_data=x_val, run_validation=True)

    torch.save(model.state_dict(), args.trainingdata[:-4] + ".pkl")
    with open(args.trainingdata[:-4] + 'dill' + '.pkl', 'wb') as f:
        serialized_model = dill.dumps(model)
        f.write(serialized_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainingdata", required=True, help="CloudTrail CSV")
    parser.add_argument("--valdata", required=True, help="CloudTrail CSV")

    main(parser.parse_args())
