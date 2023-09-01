#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import argparse
import os
from os.path import dirname
from os.path import exists

import s3fs

AZURE_TRAINING_FILES = [
    'AZUREAD_2022-08-01T00_03_56.207Z.json',
    'AZUREAD_2022-08-01T03_05_08.046Z.json',
    'AZUREAD_2022-08-01T06_16_33.925Z.json',
    'AZUREAD_2022-08-01T09_03_21.665Z.json',
    'AZUREAD_2022-08-01T12_18_15.395Z.json',
    'AZUREAD_2022-08-01T15_05_33.622Z.json',
    'AZUREAD_2022-08-01T18_01_39.553Z.json',
    'AZUREAD_2022-08-01T21_36_12.642Z.json',
    'AZUREAD_2022-08-02T00_03_57.781Z.json',
    'AZUREAD_2022-08-02T03_14_51.398Z.json',
    'AZUREAD_2022-08-02T06_08_43.120Z.json',
    'AZUREAD_2022-08-02T09_04_46.164Z.json',
    'AZUREAD_2022-08-02T12_16_42.973Z.json',
    'AZUREAD_2022-08-02T15_01_57.753Z.json',
    'AZUREAD_2022-08-02T18_07_04.839Z.json',
    'AZUREAD_2022-08-02T21_04_39.405Z.json',
    'AZUREAD_2022-08-03T00_10_42.770Z.json',
    'AZUREAD_2022-08-03T03_04_39.062Z.json',
    'AZUREAD_2022-08-03T07_03_02.647Z.json',
    'AZUREAD_2022-08-03T09_12_42.431Z.json',
    'AZUREAD_2022-08-03T12_01_59.616Z.json',
    'AZUREAD_2022-08-03T15_04_36.967Z.json',
    'AZUREAD_2022-08-03T18_33_53.602Z.json',
    'AZUREAD_2022-08-03T21_10_53.763Z.json',
    'AZUREAD_2022-08-04T00_47_51.564Z.json',
    'AZUREAD_2022-08-04T03_29_10.364Z.json',
    'AZUREAD_2022-08-04T06_22_30.326Z.json',
    'AZUREAD_2022-08-04T09_01_47.489Z.json',
    'AZUREAD_2022-08-04T12_00_37.255Z.json',
    'AZUREAD_2022-08-04T15_06_58.553Z.json',
    'AZUREAD_2022-08-04T18_20_25.773Z.json',
    'AZUREAD_2022-08-04T21_13_42.613Z.json',
    'AZUREAD_2022-08-05T00_14_48.503Z.json',
    'AZUREAD_2022-08-05T03_27_16.392Z.json',
    'AZUREAD_2022-08-05T06_14_02.065Z.json',
    'AZUREAD_2022-08-05T09_19_35.102Z.json',
    'AZUREAD_2022-08-05T12_24_54.388Z.json',
    'AZUREAD_2022-08-05T15_02_19.596Z.json',
    'AZUREAD_2022-08-05T18_07_31.442Z.json',
    'AZUREAD_2022-08-05T21_10_10.626Z.json',
    'AZUREAD_2022-08-06T00_08_48.348Z.json',
    'AZUREAD_2022-08-06T03_00_47.733Z.json',
    'AZUREAD_2022-08-06T06_00_32.252Z.json',
    'AZUREAD_2022-08-06T09_21_44.486Z.json',
    'AZUREAD_2022-08-06T12_06_11.372Z.json',
    'AZUREAD_2022-08-06T15_00_53.066Z.json',
    'AZUREAD_2022-08-06T18_15_37.469Z.json',
    'AZUREAD_2022-08-06T21_04_29.994Z.json',
    'AZUREAD_2022-08-07T00_00_23.959Z.json',
    'AZUREAD_2022-08-07T03_11_01.088Z.json',
    'AZUREAD_2022-08-07T06_02_39.472Z.json',
    'AZUREAD_2022-08-07T09_05_02.341Z.json',
    'AZUREAD_2022-08-07T12_02_57.483Z.json',
    'AZUREAD_2022-08-07T16_00_01.986Z.json',
    'AZUREAD_2022-08-07T18_27_37.071Z.json',
    'AZUREAD_2022-08-07T21_28_01.308Z.json',
    'AZUREAD_2022-08-08T00_16_13.439Z.json',
    'AZUREAD_2022-08-08T03_00_06.848Z.json',
    'AZUREAD_2022-08-08T06_10_54.304Z.json',
    'AZUREAD_2022-08-08T09_07_13.422Z.json',
    'AZUREAD_2022-08-08T12_02_00.653Z.json',
    'AZUREAD_2022-08-08T15_09_32.434Z.json',
    'AZUREAD_2022-08-08T18_05_08.444Z.json',
    'AZUREAD_2022-08-08T21_22_25.239Z.json',
    'AZUREAD_2022-08-09T00_23_17.790Z.json',
    'AZUREAD_2022-08-09T03_09_35.759Z.json',
    'AZUREAD_2022-08-09T06_01_19.285Z.json',
    'AZUREAD_2022-08-09T09_17_46.111Z.json',
    'AZUREAD_2022-08-09T12_04_05.262Z.json',
    'AZUREAD_2022-08-09T15_07_52.196Z.json',
    'AZUREAD_2022-08-09T18_22_41.664Z.json',
    'AZUREAD_2022-08-09T21_08_50.818Z.json',
    'AZUREAD_2022-08-10T00_16_26.105Z.json',
    'AZUREAD_2022-08-10T03_08_34.627Z.json',
    'AZUREAD_2022-08-10T06_18_05.500Z.json',
    'AZUREAD_2022-08-10T09_01_38.710Z.json',
    'AZUREAD_2022-08-10T12_00_06.306Z.json',
    'AZUREAD_2022-08-10T15_22_39.506Z.json',
    'AZUREAD_2022-08-10T18_02_01.694Z.json',
    'AZUREAD_2022-08-10T21_25_24.068Z.json',
    'AZUREAD_2022-08-11T00_07_04.185Z.json',
    'AZUREAD_2022-08-11T03_03_44.313Z.json',
    'AZUREAD_2022-08-11T06_20_48.479Z.json',
    'AZUREAD_2022-08-11T09_14_40.692Z.json',
    'AZUREAD_2022-08-11T12_06_40.970Z.json',
    'AZUREAD_2022-08-11T15_00_12.795Z.json',
    'AZUREAD_2022-08-11T18_04_26.402Z.json',
    'AZUREAD_2022-08-11T21_03_53.754Z.json',
    'AZUREAD_2022-08-12T00_04_59.094Z.json',
    'AZUREAD_2022-08-12T03_02_32.601Z.json',
    'AZUREAD_2022-08-12T06_20_35.849Z.json',
    'AZUREAD_2022-08-12T09_21_57.950Z.json',
    'AZUREAD_2022-08-12T12_00_16.796Z.json',
    'AZUREAD_2022-08-12T15_00_32.506Z.json',
    'AZUREAD_2022-08-12T18_00_10.559Z.json',
    'AZUREAD_2022-08-12T21_05_00.919Z.json',
    'AZUREAD_2022-08-13T00_17_54.530Z.json',
    'AZUREAD_2022-08-13T03_26_09.056Z.json',
    'AZUREAD_2022-08-13T06_00_54.755Z.json',
    'AZUREAD_2022-08-13T09_28_02.146Z.json',
    'AZUREAD_2022-08-13T12_36_25.676Z.json',
    'AZUREAD_2022-08-13T15_07_29.394Z.json',
    'AZUREAD_2022-08-13T18_43_43.943Z.json',
    'AZUREAD_2022-08-13T21_02_34.964Z.json',
    'AZUREAD_2022-08-14T00_35_43.377Z.json',
    'AZUREAD_2022-08-14T03_05_34.273Z.json',
    'AZUREAD_2022-08-14T06_29_54.324Z.json',
    'AZUREAD_2022-08-14T09_11_35.224Z.json',
    'AZUREAD_2022-08-14T12_22_29.216Z.json',
    'AZUREAD_2022-08-14T15_21_13.429Z.json',
    'AZUREAD_2022-08-14T18_12_20.996Z.json',
    'AZUREAD_2022-08-14T21_00_53.772Z.json',
    'AZUREAD_2022-08-15T00_12_27.596Z.json',
    'AZUREAD_2022-08-15T03_21_24.074Z.json',
    'AZUREAD_2022-08-15T06_05_11.987Z.json',
    'AZUREAD_2022-08-15T09_03_38.276Z.json',
    'AZUREAD_2022-08-15T12_00_09.975Z.json',
    'AZUREAD_2022-08-15T15_01_22.571Z.json',
    'AZUREAD_2022-08-15T18_04_07.706Z.json',
    'AZUREAD_2022-08-15T21_16_21.836Z.json',
    'AZUREAD_2022-08-16T00_17_08.148Z.json',
    'AZUREAD_2022-08-16T03_04_14.916Z.json',
    'AZUREAD_2022-08-16T06_00_05.126Z.json',
    'AZUREAD_2022-08-16T09_20_00.713Z.json',
    'AZUREAD_2022-08-16T12_21_42.855Z.json',
    'AZUREAD_2022-08-16T15_07_58.824Z.json',
    'AZUREAD_2022-08-16T18_26_19.807Z.json',
    'AZUREAD_2022-08-16T21_10_37.846Z.json',
    'AZUREAD_2022-08-17T00_07_39.058Z.json',
    'AZUREAD_2022-08-17T03_00_28.451Z.json',
    'AZUREAD_2022-08-17T06_13_41.197Z.json',
    'AZUREAD_2022-08-17T09_21_28.995Z.json',
    'AZUREAD_2022-08-17T12_09_04.770Z.json',
    'AZUREAD_2022-08-17T15_18_32.828Z.json',
    'AZUREAD_2022-08-17T18_02_43.590Z.json',
    'AZUREAD_2022-08-17T21_03_50.905Z.json',
    'AZUREAD_2022-08-18T00_04_04.684Z.json',
    'AZUREAD_2022-08-18T03_36_06.261Z.json',
    'AZUREAD_2022-08-18T06_02_47.638Z.json',
    'AZUREAD_2022-08-18T09_39_38.604Z.json',
    'AZUREAD_2022-08-18T12_17_15.899Z.json',
    'AZUREAD_2022-08-18T15_38_44.291Z.json',
    'AZUREAD_2022-08-18T18_16_39.557Z.json',
    'AZUREAD_2022-08-18T21_01_07.323Z.json',
    'AZUREAD_2022-08-19T00_03_26.920Z.json',
    'AZUREAD_2022-08-19T03_05_56.636Z.json',
    'AZUREAD_2022-08-19T06_15_54.060Z.json',
    'AZUREAD_2022-08-19T09_19_05.120Z.json',
    'AZUREAD_2022-08-19T12_17_08.196Z.json',
    'AZUREAD_2022-08-19T15_15_11.004Z.json',
    'AZUREAD_2022-08-19T18_01_30.625Z.json',
    'AZUREAD_2022-08-19T21_01_20.621Z.json',
    'AZUREAD_2022-08-20T00_19_23.348Z.json',
    'AZUREAD_2022-08-20T03_04_23.422Z.json',
    'AZUREAD_2022-08-20T06_02_30.851Z.json',
    'AZUREAD_2022-08-20T09_23_15.870Z.json',
    'AZUREAD_2022-08-20T12_02_07.076Z.json',
    'AZUREAD_2022-08-20T15_02_02.602Z.json',
    'AZUREAD_2022-08-20T18_41_15.371Z.json',
    'AZUREAD_2022-08-20T21_04_09.868Z.json',
    'AZUREAD_2022-08-21T00_21_03.378Z.json',
    'AZUREAD_2022-08-21T03_01_03.612Z.json',
    'AZUREAD_2022-08-21T06_31_46.721Z.json',
    'AZUREAD_2022-08-21T09_00_25.563Z.json',
    'AZUREAD_2022-08-21T12_00_28.297Z.json',
    'AZUREAD_2022-08-21T15_06_02.955Z.json',
    'AZUREAD_2022-08-21T18_02_36.810Z.json',
    'AZUREAD_2022-08-21T21_11_47.527Z.json',
    'AZUREAD_2022-08-22T00_38_39.339Z.json',
    'AZUREAD_2022-08-22T03_12_35.427Z.json',
    'AZUREAD_2022-08-22T06_10_22.996Z.json',
    'AZUREAD_2022-08-22T09_01_26.005Z.json',
    'AZUREAD_2022-08-22T12_30_36.375Z.json',
    'AZUREAD_2022-08-22T15_11_57.786Z.json',
    'AZUREAD_2022-08-22T18_06_12.318Z.json',
    'AZUREAD_2022-08-22T21_06_16.397Z.json',
    'AZUREAD_2022-08-23T00_01_32.097Z.json',
    'AZUREAD_2022-08-23T03_13_34.617Z.json',
    'AZUREAD_2022-08-23T06_12_04.524Z.json',
    'AZUREAD_2022-08-23T09_06_36.465Z.json',
    'AZUREAD_2022-08-23T12_23_47.260Z.json',
    'AZUREAD_2022-08-23T15_07_25.933Z.json',
    'AZUREAD_2022-08-23T18_06_17.979Z.json',
    'AZUREAD_2022-08-23T21_10_23.207Z.json',
    'AZUREAD_2022-08-24T00_17_52.726Z.json',
    'AZUREAD_2022-08-24T03_08_04.379Z.json',
    'AZUREAD_2022-08-24T06_12_36.113Z.json',
    'AZUREAD_2022-08-24T09_02_01.941Z.json',
    'AZUREAD_2022-08-24T12_05_38.515Z.json',
    'AZUREAD_2022-08-24T15_00_59.959Z.json',
    'AZUREAD_2022-08-24T18_16_26.757Z.json',
    'AZUREAD_2022-08-24T21_01_57.951Z.json',
    'AZUREAD_2022-08-25T00_19_39.558Z.json',
    'AZUREAD_2022-08-25T03_00_11.988Z.json',
    'AZUREAD_2022-08-25T06_09_30.965Z.json',
    'AZUREAD_2022-08-25T09_06_29.348Z.json',
    'AZUREAD_2022-08-25T12_21_23.282Z.json',
    'AZUREAD_2022-08-25T15_18_51.497Z.json',
    'AZUREAD_2022-08-25T18_08_51.671Z.json',
    'AZUREAD_2022-08-25T21_16_57.141Z.json',
    'AZUREAD_2022-08-26T00_04_23.818Z.json',
    'AZUREAD_2022-08-26T03_35_07.038Z.json',
    'AZUREAD_2022-08-26T06_09_07.095Z.json',
    'AZUREAD_2022-08-26T09_13_45.158Z.json',
    'AZUREAD_2022-08-26T12_04_24.131Z.json',
    'AZUREAD_2022-08-26T15_04_03.469Z.json',
    'AZUREAD_2022-08-26T18_11_20.205Z.json',
    'AZUREAD_2022-08-26T21_10_28.387Z.json',
    'AZUREAD_2022-08-27T00_06_18.712Z.json',
    'AZUREAD_2022-08-27T03_04_39.876Z.json',
    'AZUREAD_2022-08-27T06_25_32.944Z.json',
    'AZUREAD_2022-08-27T09_11_24.479Z.json',
    'AZUREAD_2022-08-27T12_15_05.051Z.json',
    'AZUREAD_2022-08-27T15_13_13.225Z.json',
    'AZUREAD_2022-08-27T18_03_47.921Z.json',
    'AZUREAD_2022-08-27T21_09_47.684Z.json',
    'AZUREAD_2022-08-28T00_06_04.882Z.json',
    'AZUREAD_2022-08-28T03_11_50.094Z.json',
    'AZUREAD_2022-08-28T06_22_12.988Z.json',
    'AZUREAD_2022-08-28T09_06_53.959Z.json',
    'AZUREAD_2022-08-28T12_34_13.560Z.json',
    'AZUREAD_2022-08-28T15_01_28.549Z.json',
    'AZUREAD_2022-08-28T18_00_44.018Z.json',
    'AZUREAD_2022-08-28T21_13_56.458Z.json',
    'AZUREAD_2022-08-29T00_13_43.369Z.json',
    'AZUREAD_2022-08-29T03_45_56.645Z.json',
    'AZUREAD_2022-08-29T06_27_25.048Z.json',
    'AZUREAD_2022-08-29T09_21_38.347Z.json',
    'AZUREAD_2022-08-29T12_05_54.323Z.json',
    'AZUREAD_2022-08-29T15_02_31.484Z.json',
    'AZUREAD_2022-08-29T18_06_13.069Z.json',
    'AZUREAD_2022-08-29T21_21_41.645Z.json'
]

AZURE_INFERENCE_FILES = [
    'AZUREAD_2022-08-30T00_17_05.561Z.json',
    'AZUREAD_2022-08-30T03_14_27.626Z.json',
    'AZUREAD_2022-08-30T06_15_21.422Z.json',
    'AZUREAD_2022-08-30T09_21_58.312Z.json',
    'AZUREAD_2022-08-30T12_05_53.775Z.json',
    'AZUREAD_2022-08-30T15_05_34.679Z.json',
    'AZUREAD_2022-08-30T18_39_54.214Z.json',
    'AZUREAD_2022-08-30T21_01_48.448Z.json',
    'AZUREAD_2022-08-31T00_21_46.153Z.json',
    'AZUREAD_2022-08-31T03_08_27.951Z.json',
    'AZUREAD_2022-08-31T06_20_09.178Z.json',
    'AZUREAD_2022-08-31T09_01_27.089Z.json',
    'AZUREAD_2022-08-31T12_02_02.230Z.json',
    'AZUREAD_2022-08-31T15_03_06.756Z.json',
    'AZUREAD_2022-08-31T18_03_06.102Z.json',
    'AZUREAD_2022-08-31T21_13_44.759Z.json'
]

DUO_TRAINING_FILES = [
    'DUO_2022-08-01T00_05_06.806Z.json',
    'DUO_2022-08-01T03_02_04.418Z.json',
    'DUO_2022-08-01T06_05_05.064Z.json',
    'DUO_2022-08-01T09_55_08.757Z.json',
    'DUO_2022-08-01T12_09_47.901Z.json',
    'DUO_2022-08-01T15_08_57.986Z.json',
    'DUO_2022-08-01T18_05_32.818Z.json',
    'DUO_2022-08-01T21_17_59.018Z.json',
    'DUO_2022-08-02T00_37_03.298Z.json',
    'DUO_2022-08-02T03_36_35.233Z.json',
    'DUO_2022-08-02T06_26_03.986Z.json',
    'DUO_2022-08-02T09_01_18.144Z.json',
    'DUO_2022-08-02T12_08_27.244Z.json',
    'DUO_2022-08-02T15_07_44.984Z.json',
    'DUO_2022-08-02T18_56_34.378Z.json',
    'DUO_2022-08-02T21_10_29.396Z.json',
    'DUO_2022-08-03T00_01_48.778Z.json',
    'DUO_2022-08-03T03_05_22.026Z.json',
    'DUO_2022-08-03T06_00_14.663Z.json',
    'DUO_2022-08-03T09_14_08.835Z.json',
    'DUO_2022-08-03T12_17_51.740Z.json',
    'DUO_2022-08-03T15_17_09.808Z.json',
    'DUO_2022-08-03T18_17_59.005Z.json',
    'DUO_2022-08-03T21_02_12.484Z.json',
    'DUO_2022-08-04T00_16_45.964Z.json',
    'DUO_2022-08-04T03_20_44.449Z.json',
    'DUO_2022-08-04T06_05_25.390Z.json',
    'DUO_2022-08-04T09_06_36.229Z.json',
    'DUO_2022-08-04T12_12_42.099Z.json',
    'DUO_2022-08-04T15_09_12.877Z.json',
    'DUO_2022-08-04T18_13_07.708Z.json',
    'DUO_2022-08-04T21_08_34.357Z.json',
    'DUO_2022-08-05T00_39_29.224Z.json',
    'DUO_2022-08-05T03_58_45.946Z.json',
    'DUO_2022-08-05T06_22_42.332Z.json',
    'DUO_2022-08-05T09_31_47.259Z.json',
    'DUO_2022-08-05T12_05_50.568Z.json',
    'DUO_2022-08-05T15_00_17.239Z.json',
    'DUO_2022-08-05T18_05_56.244Z.json',
    'DUO_2022-08-05T21_03_44.044Z.json',
    'DUO_2022-08-06T00_04_07.964Z.json',
    'DUO_2022-08-06T03_00_14.884Z.json',
    'DUO_2022-08-06T06_14_11.811Z.json',
    'DUO_2022-08-06T09_17_14.197Z.json',
    'DUO_2022-08-06T13_00_53.987Z.json',
    'DUO_2022-08-06T15_04_28.652Z.json',
    'DUO_2022-08-06T18_11_42.754Z.json',
    'DUO_2022-08-06T21_01_46.563Z.json',
    'DUO_2022-08-07T01_30_43.028Z.json',
    'DUO_2022-08-07T03_59_14.016Z.json',
    'DUO_2022-08-07T06_38_45.747Z.json',
    'DUO_2022-08-07T09_12_23.830Z.json',
    'DUO_2022-08-07T12_09_20.360Z.json',
    'DUO_2022-08-07T15_01_43.630Z.json',
    'DUO_2022-08-07T18_25_51.363Z.json',
    'DUO_2022-08-07T21_06_39.592Z.json',
    'DUO_2022-08-08T00_03_47.268Z.json',
    'DUO_2022-08-08T03_08_12.355Z.json',
    'DUO_2022-08-08T06_08_16.424Z.json',
    'DUO_2022-08-08T09_35_08.045Z.json',
    'DUO_2022-08-08T12_37_44.191Z.json',
    'DUO_2022-08-08T15_34_42.886Z.json',
    'DUO_2022-08-08T18_02_49.470Z.json',
    'DUO_2022-08-08T21_01_26.207Z.json',
    'DUO_2022-08-09T00_02_38.932Z.json',
    'DUO_2022-08-09T03_06_46.584Z.json',
    'DUO_2022-08-09T06_15_35.216Z.json',
    'DUO_2022-08-09T09_42_11.514Z.json',
    'DUO_2022-08-09T12_00_29.207Z.json',
    'DUO_2022-08-09T15_25_00.138Z.json',
    'DUO_2022-08-09T18_00_22.295Z.json',
    'DUO_2022-08-09T21_06_00.541Z.json',
    'DUO_2022-08-10T00_06_51.783Z.json',
    'DUO_2022-08-10T03_12_39.799Z.json',
    'DUO_2022-08-10T06_02_17.599Z.json',
    'DUO_2022-08-10T09_02_53.713Z.json',
    'DUO_2022-08-10T12_22_27.222Z.json',
    'DUO_2022-08-10T16_18_07.097Z.json',
    'DUO_2022-08-10T18_03_12.024Z.json',
    'DUO_2022-08-10T21_05_02.676Z.json',
    'DUO_2022-08-11T00_14_46.056Z.json',
    'DUO_2022-08-11T03_09_03.122Z.json',
    'DUO_2022-08-11T06_13_34.994Z.json',
    'DUO_2022-08-11T09_07_33.742Z.json',
    'DUO_2022-08-11T12_15_05.859Z.json',
    'DUO_2022-08-11T15_08_54.468Z.json',
    'DUO_2022-08-11T18_03_29.894Z.json',
    'DUO_2022-08-11T21_07_36.914Z.json',
    'DUO_2022-08-12T00_00_36.965Z.json',
    'DUO_2022-08-12T03_09_13.102Z.json',
    'DUO_2022-08-12T06_23_37.076Z.json',
    'DUO_2022-08-12T09_05_25.800Z.json',
    'DUO_2022-08-12T12_00_02.586Z.json',
    'DUO_2022-08-12T15_07_04.329Z.json',
    'DUO_2022-08-12T18_06_29.281Z.json',
    'DUO_2022-08-12T21_06_02.030Z.json',
    'DUO_2022-08-13T00_01_19.915Z.json',
    'DUO_2022-08-13T03_19_52.793Z.json',
    'DUO_2022-08-13T06_07_44.115Z.json',
    'DUO_2022-08-13T09_30_28.265Z.json',
    'DUO_2022-08-13T12_05_54.411Z.json',
    'DUO_2022-08-13T15_32_27.481Z.json',
    'DUO_2022-08-13T18_30_04.101Z.json',
    'DUO_2022-08-13T21_08_32.899Z.json',
    'DUO_2022-08-14T00_28_59.030Z.json',
    'DUO_2022-08-14T03_10_53.115Z.json',
    'DUO_2022-08-14T06_15_59.959Z.json',
    'DUO_2022-08-14T09_05_12.470Z.json',
    'DUO_2022-08-14T12_07_08.971Z.json',
    'DUO_2022-08-14T15_18_53.132Z.json',
    'DUO_2022-08-14T18_34_28.408Z.json',
    'DUO_2022-08-14T22_07_37.863Z.json',
    'DUO_2022-08-15T00_15_04.480Z.json',
    'DUO_2022-08-15T03_01_24.327Z.json',
    'DUO_2022-08-15T06_04_36.244Z.json',
    'DUO_2022-08-15T09_14_15.659Z.json',
    'DUO_2022-08-15T12_42_54.708Z.json',
    'DUO_2022-08-15T15_26_14.651Z.json',
    'DUO_2022-08-15T18_29_34.302Z.json',
    'DUO_2022-08-15T21_03_55.385Z.json',
    'DUO_2022-08-16T00_18_15.026Z.json',
    'DUO_2022-08-16T03_00_38.697Z.json',
    'DUO_2022-08-16T06_10_49.268Z.json',
    'DUO_2022-08-16T09_11_31.564Z.json',
    'DUO_2022-08-16T12_08_44.078Z.json',
    'DUO_2022-08-16T15_00_07.447Z.json',
    'DUO_2022-08-16T18_02_26.218Z.json',
    'DUO_2022-08-16T21_13_00.224Z.json',
    'DUO_2022-08-17T00_04_52.924Z.json',
    'DUO_2022-08-17T03_37_32.167Z.json',
    'DUO_2022-08-17T06_08_04.647Z.json',
    'DUO_2022-08-17T09_02_22.845Z.json',
    'DUO_2022-08-17T12_02_54.475Z.json',
    'DUO_2022-08-17T15_15_30.542Z.json',
    'DUO_2022-08-17T18_04_42.665Z.json',
    'DUO_2022-08-17T21_07_41.110Z.json',
    'DUO_2022-08-18T00_00_50.864Z.json',
    'DUO_2022-08-18T03_03_50.747Z.json',
    'DUO_2022-08-18T06_00_37.820Z.json',
    'DUO_2022-08-18T09_04_22.633Z.json',
    'DUO_2022-08-18T12_09_18.662Z.json',
    'DUO_2022-08-18T15_14_45.798Z.json',
    'DUO_2022-08-18T18_17_27.739Z.json',
    'DUO_2022-08-18T21_46_46.184Z.json',
    'DUO_2022-08-19T00_01_58.530Z.json',
    'DUO_2022-08-19T03_02_51.459Z.json',
    'DUO_2022-08-19T07_06_56.960Z.json',
    'DUO_2022-08-19T09_00_18.242Z.json',
    'DUO_2022-08-19T12_20_43.912Z.json',
    'DUO_2022-08-19T15_08_51.811Z.json',
    'DUO_2022-08-19T18_09_33.257Z.json',
    'DUO_2022-08-19T21_04_15.361Z.json',
    'DUO_2022-08-20T00_03_51.763Z.json',
    'DUO_2022-08-20T03_19_11.469Z.json',
    'DUO_2022-08-20T06_21_39.116Z.json',
    'DUO_2022-08-20T09_59_11.004Z.json',
    'DUO_2022-08-20T12_01_02.871Z.json',
    'DUO_2022-08-20T15_08_10.265Z.json',
    'DUO_2022-08-20T18_09_04.018Z.json',
    'DUO_2022-08-20T21_05_22.435Z.json',
    'DUO_2022-08-21T00_12_40.168Z.json',
    'DUO_2022-08-21T03_40_42.920Z.json',
    'DUO_2022-08-21T06_19_14.897Z.json',
    'DUO_2022-08-21T09_04_54.519Z.json',
    'DUO_2022-08-21T12_02_14.649Z.json',
    'DUO_2022-08-21T15_03_59.410Z.json',
    'DUO_2022-08-21T18_40_23.810Z.json',
    'DUO_2022-08-21T21_19_39.429Z.json',
    'DUO_2022-08-22T00_00_32.032Z.json',
    'DUO_2022-08-22T03_15_10.168Z.json',
    'DUO_2022-08-22T06_14_12.112Z.json',
    'DUO_2022-08-22T09_40_53.927Z.json',
    'DUO_2022-08-22T12_09_11.101Z.json',
    'DUO_2022-08-22T15_04_12.123Z.json',
    'DUO_2022-08-22T18_04_02.147Z.json',
    'DUO_2022-08-22T21_07_41.681Z.json',
    'DUO_2022-08-23T00_45_51.610Z.json',
    'DUO_2022-08-23T03_15_17.210Z.json',
    'DUO_2022-08-23T06_09_53.853Z.json',
    'DUO_2022-08-23T09_03_35.671Z.json',
    'DUO_2022-08-23T12_20_07.116Z.json',
    'DUO_2022-08-23T15_19_13.637Z.json',
    'DUO_2022-08-23T18_23_01.179Z.json',
    'DUO_2022-08-23T21_30_50.797Z.json',
    'DUO_2022-08-24T00_35_12.724Z.json',
    'DUO_2022-08-24T03_07_48.905Z.json',
    'DUO_2022-08-24T06_52_26.658Z.json',
    'DUO_2022-08-24T09_02_32.667Z.json',
    'DUO_2022-08-24T12_36_29.646Z.json',
    'DUO_2022-08-24T15_14_52.489Z.json',
    'DUO_2022-08-24T18_10_31.894Z.json',
    'DUO_2022-08-24T21_23_58.385Z.json',
    'DUO_2022-08-25T00_38_52.875Z.json',
    'DUO_2022-08-25T03_09_40.392Z.json',
    'DUO_2022-08-25T06_16_01.732Z.json',
    'DUO_2022-08-25T09_15_31.540Z.json',
    'DUO_2022-08-25T12_32_54.214Z.json',
    'DUO_2022-08-25T15_04_51.656Z.json',
    'DUO_2022-08-25T18_09_25.686Z.json',
    'DUO_2022-08-25T21_30_36.994Z.json',
    'DUO_2022-08-26T00_03_13.149Z.json',
    'DUO_2022-08-26T03_02_51.844Z.json',
    'DUO_2022-08-26T06_15_36.205Z.json',
    'DUO_2022-08-26T09_12_38.216Z.json',
    'DUO_2022-08-26T12_18_33.577Z.json',
    'DUO_2022-08-26T15_12_34.213Z.json',
    'DUO_2022-08-26T18_26_38.686Z.json',
    'DUO_2022-08-26T21_00_41.408Z.json',
    'DUO_2022-08-27T00_17_07.536Z.json',
    'DUO_2022-08-27T03_08_55.907Z.json',
    'DUO_2022-08-27T06_01_59.976Z.json',
    'DUO_2022-08-27T10_04_58.247Z.json',
    'DUO_2022-08-27T12_00_48.542Z.json',
    'DUO_2022-08-27T15_03_54.319Z.json',
    'DUO_2022-08-27T18_04_17.126Z.json',
    'DUO_2022-08-27T21_07_43.779Z.json',
    'DUO_2022-08-28T00_26_28.894Z.json',
    'DUO_2022-08-28T04_04_08.201Z.json',
    'DUO_2022-08-28T06_15_53.091Z.json',
    'DUO_2022-08-28T09_33_01.613Z.json',
    'DUO_2022-08-28T12_38_30.312Z.json',
    'DUO_2022-08-28T15_38_27.355Z.json',
    'DUO_2022-08-28T18_03_05.686Z.json',
    'DUO_2022-08-28T21_10_45.915Z.json',
    'DUO_2022-08-29T00_07_32.403Z.json',
    'DUO_2022-08-29T03_51_25.190Z.json',
    'DUO_2022-08-29T06_06_27.003Z.json',
    'DUO_2022-08-29T09_26_52.838Z.json',
    'DUO_2022-08-29T12_09_06.633Z.json',
    'DUO_2022-08-29T15_00_36.208Z.json',
    'DUO_2022-08-29T18_02_32.315Z.json',
    'DUO_2022-08-29T21_10_08.476Z.json'
]

DUO_INFERENCE_FILES = [
    'DUO_2022-08-30T00_26_18.130Z.json',
    'DUO_2022-08-30T03_33_23.301Z.json',
    'DUO_2022-08-30T06_38_13.972Z.json',
    'DUO_2022-08-30T09_14_52.103Z.json',
    'DUO_2022-08-30T12_12_56.353Z.json',
    'DUO_2022-08-30T15_42_24.545Z.json',
    'DUO_2022-08-30T18_35_34.524Z.json',
    'DUO_2022-08-30T21_02_48.060Z.json',
    'DUO_2022-08-31T00_01_31.947Z.json',
    'DUO_2022-08-31T03_06_14.133Z.json',
    'DUO_2022-08-31T06_04_50.555Z.json',
    'DUO_2022-08-31T09_04_50.225Z.json',
    'DUO_2022-08-31T12_00_48.690Z.json',
    'DUO_2022-08-31T15_00_12.020Z.json',
    'DUO_2022-08-31T18_00_01.228Z.json',
    'DUO_2022-08-31T21_01_09.338Z.json'
]

DFP_DATASET_FILES = {
    'azure': (AZURE_TRAINING_FILES, AZURE_INFERENCE_FILES), 'duo': (DUO_TRAINING_FILES, DUO_INFERENCE_FILES)
}

S3_BASE_PATH = "/rapidsai-data/cyber/morpheus/dfp/"
EXAMPLE_DATA_DIR = dirname(dirname(os.path.abspath(__file__))) + "/data"


def fetch_dataset(dataset):

    ds_filenames = DFP_DATASET_FILES[dataset]
    fs_hndl = s3fs.S3FileSystem(anon=True)
    s3_base_path = os.path.join(S3_BASE_PATH, dataset)

    train_dir = f"{EXAMPLE_DATA_DIR}/dfp/{dataset}-training-data/"
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    train_files = ds_filenames[0]
    for f in train_files:
        if not exists(train_dir + f):
            print(f"Downloading {f}")
            fs_hndl.get_file(os.path.join(s3_base_path, f), train_dir + f)

    infer_dir = f"{EXAMPLE_DATA_DIR}/dfp/{dataset}-inference-data/"
    if not exists(infer_dir):
        os.makedirs(infer_dir)

    infer_files = ds_filenames[1]
    for f in infer_files:
        if not os.path.exists(infer_dir + f):
            print(f"Downloading {f}")
            fs_hndl.get_file(os.path.join(s3_base_path, f), infer_dir + f)


def parse_args():
    argparser = argparse.ArgumentParser("Fetches training and inference data for DFP examples")
    argparser.add_argument("data_set", nargs='*', choices=['azure', 'duo', 'all'], help="Data set to fetch")

    args = argparser.parse_args()

    return args


def main():
    args = parse_args()
    if args.data_set == ['all']:
        ds_list = DFP_DATASET_FILES.keys()
    else:
        ds_list = args.data_set

    for dataset in ds_list:
        fetch_dataset(dataset)


if __name__ == "__main__":
    main()
