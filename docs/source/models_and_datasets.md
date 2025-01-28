<!--
SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Morpheus Models

Morpheus comes with a number of pre-trained models with corresponding training, validation scripts, and datasets. The latest release of these models can be found [here](https://github.com/nv-morpheus/Morpheus/blob/-/models).

|Model|GPU Mem Req|Description|
|-----|-----------|-----------|
|Anomalous Behavior Profiling (ABP)|2015MiB|This model is an example of a binary classifier to differentiate between anomalous GPU behavior such as cryptocurrency mining / GPU malware, and non-anomalous GPU-based workflows (for example, ML/DL training). The model is an XGBoost model.|
|Digital Fingerprinting (DFP)|4.97MiB|This use case is currently implemented to detect changes in a users' behavior that indicates a change from a human to a machine or a machine to a human. The model is an ensemble of an Autoencoder and fast Fourier transform reconstruction.|
|Fraud Detection|76.55MiB|This model shows an application of a graph neural network for fraud detection in a credit card transaction graph. A transaction dataset that includes three types of nodes, transaction, client, and merchant nodes is used for modeling. A combination of [GraphSAGE](https://snap.stanford.edu/graphsage/) along with [XGBoost](https://xgboost.readthedocs.io/en/stable/) is used to identify frauds in the transaction networks.|
|Ransomware Detection Model|n/a|This model shows an application of DOCA AppShield to use data from volatile memory to classify processes as ransomware or benign. This model uses a sliding window over time and feeds derived data into a random forest classifiers of various lengths depending on the amount of data collected.|
|Flexible Log Parsing|1612MiB|This model is an example of using Named Entity Recognition (NER) for log parsing, specifically [Apache HTTP Server](https://httpd.apache.org/) logs.|
