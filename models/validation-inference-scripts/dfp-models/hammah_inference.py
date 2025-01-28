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
python hammah_inference.py --validationdata \
    ../../datasets/validation-data/dfp-cloudtrail-user123-validation-data-input.csv \
 --model ../../training-tuning-scripts/dfp-models/hammah-user123-20211017dill.pkl \
      --output abp-validation-output.csv

"""

import argparse
import datetime

import cupy as cp
import dill
import pandas as pd
import periodicity_detection as pdd

import cudf


def infer(validationdata, model, output):

    def zscore(data):
        mean = cp.mean(data)
        std = cp.std(data)
        return (data - mean) / std

    def date2min(time):
        start = start_time
        timesince = time - start
        return int(timesince.total_seconds() // 60)

    form = "%Y-%m-%dT%H:%M:%SZ"

    def stript(input_str):
        obj = datetime.datetime.strptime(input_str, form)
        return obj

    def back_to_string(obj):
        # return "{}-{}:{}:{}".format(f"{obj.month:02}", f"{obj.day:02}", f"{obj.hour:02}", f"{obj.minute:02}")
        return f"{obj.month:02}-{obj.day:02}-{obj.hour:02}-{obj.minute:02}"

    x_validation = pd.read_csv(validationdata)

    col_list = [
        'userIdentityaccountId',
        'eventSource',
        'eventName',
        'sourceIPAddress',
        'userAgent',
        'userIdentitytype',
        'apiVersion',
        'userIdentityprincipalId',
        'userIdentityarn',
        'userIdentityaccessKeyId',
        'tlsDetailsclientProvidedHostHeader',
        'errorCode',
        'errorMessage',
        'userIdentitysessionContextsessionIssueruserName'
    ]

    for i in list(x_validation):
        if i not in col_list:
            x_validation = x_validation.drop(i, axis=1)

    with open(model, 'rb') as f:
        model = dill.load(f)

    scores = model.get_anomaly_score(x_validation)[3]
    x_validation['ae_anomaly_score'] = scores

    x_validation.sort_values('ae_anomaly_score', ascending=False).head(10)
    # since inference is done, add the original columns back so the output will be the same as the input format
    # X_val['ts_anomaly']=X_val_original['ts_anomaly']
    df = cudf.read_csv(validationdata)
    df = df.sort_values(by=['eventTime'])
    timearr = df.eventTime.to_numpy()
    start_time = stript(timearr[0])
    timeobj = list(map(stript, timearr))
    h_s = list(map(date2min, timeobj))
    num, _ = cp.histogram(cp.array(h_s), bins=cp.arange(0, max(h_s)))
    signal = cudf.Series(num)
    # was 'a' before pylint
    a_vector = cp.fromDlpack(signal.to_dlpack())
    periodogram = pdd.to_periodogram(signal)
    periodogram = periodogram[:int((len(signal) / 2))]
    threshold = float(cp.percentile(cp.array(periodogram), 90))
    indices = cudf.Series(cp.arange(len(periodogram)))[periodogram < threshold].to_numpy()
    rft = cp.fft.rfft(a_vector)
    rft[indices] = 0
    recon = cp.fft.irfft(rft)
    err = (abs(recon - a_vector))
    z_score_value = zscore(err)
    indices = cudf.Series(cp.arange(len(z_score_value)))[z_score_value >= 8].to_numpy()
    strlist = []
    for mins in indices:
        from_start = start_time + datetime.timedelta(minutes=int(mins))
        strlist.append(back_to_string(from_start))
    df['ts_anomaly'] = False
    for i in strlist:
        df['ts_anomaly'] = df['eventTime'].str.contains(i)
    x_validation.insert(0, 'eventID', x_validation.index)
    x_validation.insert(0, '', x_validation.index)
    x_validation['ts_anomaly'] = df['ts_anomaly'].to_pandas()
    x_validation.to_csv(output, index=False)


def main(args):

    infer(args.validationdata, args.model, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validationdata", required=True, help="Labelled data in JSON format")
    parser.add_argument("--model", required=True, help="trained model")
    parser.add_argument("--output", required=True, help="output filename")

    main(parser.parse_args())
