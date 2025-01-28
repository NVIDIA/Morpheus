<!--
SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

Pretrained models for Morpheus with corresponding training, validation scripts, and datasets.

## Repo Structure
Every Morpheus use case has a directory, **`<use-case>-models`**, that contains the model files for the use case. Training and validation datasets and scripts are also provided in [`datasets`](./datasets/), [`training-tuning-scripts`](./training-tuning-scripts/), and [`validation-inference-scripts`](./validation-inference-scripts/). Jupyter notebook (`.ipynb`) version of the training and fine-tuning scripts are also provided.

The `triton_model_repo` contains the necessary directory structure and configuration files in order to run the Morpheus Models in Triton Inference Server. This includes symlinks to the above-mentioned model files along with corresponding Triton configuration files (`.pbtxt`). More information on how to deploy this repository to Triton can be found in the [README](./triton-model-repo/README.md).

Models can also be published to an [MLflow](https://mlflow.org/) server and deployed to Triton using the [MLflow Triton plugin](https://github.com/triton-inference-server/server/tree/main/deploy/mlflow-triton-plugin). The [MLflow](./mlflow/README.md) directory contains information on how to set up a Docker container to run an MLflow server for publishing Morpheus models and deploying them to Triton.

In the root directory, the file `model-information.csv` contains the following information for each model:

 - **Model name** - Name of the model
 - **Use case** - Specific Morpheus use case the model targets
 - **Owner** - Name of the individual who owns the model
 - **Version** - Version of the model (major.minor.patch)
 - **Model overview** - General description
 - **Model architecture** - General model architecture
 - **Training** - Training dataset and paradigm
 - **How to use this model** - Circumstances where this model is useful
 - **Input data** - Typical data that is used as input to the model
 - **Output** - Type and format of model output
 - **Out-of-scope use cases** - Use cases not envisioned during development
 - **Ethical considerations** - Ethical analysis of risks and harms
 - **References** - Resources used in model development
 - **Training epochs** - Number of epochs used during training
 - **Batch size** - Batch size used during training
 - **GPU model** - Family of GPU used during training
 - **Model accuracy** - Accuracy of the model when tested
 - **Model F1** - F1 score of the model when tested
 - **Small test set accuracy** - Accuracy of model on validation data in datasets directory
 - **Memory footprint** - Memory required by the model
 - **Thresholds** - Values of thresholds used for validation
 - **NLP hash file** - Hash file for tokenizer vocabulary
 - **NLP max length** - Max length value for tokenizer
 - **NLP stride** - stride value for tokenizer
 - **NLP do lower** - `do_lower` value for tokenizer
 - **NLP do truncate** - `do_truncate` value for tokenizer
 - **Version CUDA** - CUDA version used during training
 - **Version Python** - Python version used during training
 - **Version Ubuntu** - Ubuntu version used during training
 - **Version Transformers** - Transformers version used during training

## Generating TensorRT Models from ONNX
The Morpheus ONNX to TensorRT (TRT) conversion utility requires additional packages, which can be installed using the following command:
```bash
conda env update --solver=libmamba -n morpheus --file conda/environments/model-utils_cuda-125_arch-$(arch).yaml
```

For users wishing to use TRT models generated using this tool with Triton will then need to refer to the [Triton Model Repository](./triton-model-repo/README.md) documentation on how to deploy the models to Triton, and how to build the Morpheus Triton Server Models Container.

# Model Card Info
## Sensitive Information Detection (SID)
### Model Overview
SID is a classifier, designed to detect sensitive information (for example, AWS credentials, GitHub credentials) in unencrypted data. This example model classifies text containing these 10 categories of sensitive information- address, bank account, credit card number, email address, government id number, full name, password, phone number, secret keys, and usernames.
### Model Architecture
Compact BERT-mini transformer model
### Training
Training consisted of fine-tuning the original pretrained [model from Google](https://huggingface.co/google/bert_uncased_L-4_H-256_A-4). The labeled training dataset is 2 million synthetic PCAP payloads generated using the [faker package](https://github.com/joke2k/faker) to mimic sensitive and benign data found in nested JSON objects from web APIs and environmental variables.
### How To Use This Model
This model is an example of customized transformer-based sensitive information detection. It can be further fine-tuned for specific detection needs or retrained for alternative categorizations using the fine-tuning scripts in the repo.
#### Input
English text from PCAP payloads
#### Output
Multi-label sequence classification for 10 sensitive information categories
### Generating TRT Models from ONNX

For the best performance you need to compile a TensorRT engine file on each machine that it will be run on. To facilitate this, Morpheus contains a utility to input an ONNX file and export the TensorRT engine file. Sample command to generate the TensorRT engine file -
```bash
morpheus --log_level=info tools onnx-to-trt --input_model ${MORPHEUS_ROOT}/models/sid-models/sid-minibert-20230424.onnx --output_model ${MORPHEUS_ROOT}/models/sid-models/model.plan --batches 1 8 --batches 1 16 --batches 1 32 --seq_length 256 --max_workspace_size 16000
```

Note: If you get an out-of-memory error, reduce the `--max_workspace_size` argument until it will successfully run.
### References
Well-Read Students Learn Better: On the Importance of Pre-training Compact Models, 2019,  https://arxiv.org/abs/1908.08962

## Phishing Email Detection
### Model Overview
Phishing email detection is a binary classifier differentiating between phishing/spam and non-phishing/spam emails and SMS messages.
### Model Architecture
BERT-base uncased transformer model
### Training
Training consisted of fine-tuning the original pretrained [model from google](https://huggingface.co/bert-base-uncased). The labeled training dataset is around 5000 SMS messages from a public dataset- [SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
### How To Use This Model
This model is an example of customized transformer-based phishing email detection. It can be retrained for specific detection needs and customized the emails of your enterprise using the training scripts in the repo.
#### Input
Entire email as a string
#### Output
Binary sequence classification as phishing/spam or non-phishing/spam
### Generating TRT Models from ONNX
For the best performance you need to compile a TensorRT engine file on each machine that it will be run on. To facilitate this, Morpheus contains a utility to input an ONNX file and export the TensorRT engine file. Sample command to generate the TensorRT engine file -
```bash
morpheus --log_level=info tools onnx-to-trt --input_model ${MORPHEUS_ROOT}/models/phishing-models/phishing-bert-20230517.onnx --output_model ${MORPHEUS_ROOT}/models/phishing-models/model.plan --batches 1 8 --batches 1 16 --batches 1 32 --seq_length 128 --max_workspace_size 16000
```
### References
- https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
- Devlin J. et al. (2018), BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
https://arxiv.org/abs/1810.04805


## Anomalous Behavior Profiling (ABP)
### Model Overview
This model is an example of a binary classifier to differentiate between anomalous GPU behavior such as cryptocurrency mining / GPU malware, and non-anomalous GPU-based workflows (for example, ML/DL training). The model is an XGBoost model.
### Model Architecture
XGBoost
### Training
Training consisted of ~1000 labeled nv-smi logs generated from processes running either GPU malware or benign GPU-based workflows.
### How To Use This Model
This model can be used to flag anomalous GPU activity.
#### Input
nv-smi data
#### Output
Binary classification as anomalous or benign.
### References
<!-- Vale attempts to spell check the author names -->
<!-- vale off -->
Chen, Guestrin (2016) XGBoost. A scalable tree boosting system. https://arxiv.org/abs/1603.02754
<!-- vale on -->

## Digital Fingerprinting (DFP)
### Model Overview
This use case is currently implemented to detect changes in users' behavior that indicate a change from a human to a machine or a machine to a human. The model is an ensemble of an Autoencoder and fast Fourier transform reconstruction.
### Model Architecture
The model is an ensemble of an Autoencoder and a fast Fourier transform reconstruction. The reconstruction loss of new log data through the trained Autoencoder is used as an anomaly score. Concurrently, the timestamps of user/entity activity are used for a time series analysis to flag activity with poor reconstruction after a fast Fourier transform.
### Training
The Autoencoder is trained on a baseline benign period of user activity.
### How To Use This Model
This model is one example of an Autoencoder trained from a baseline for benign activity from synthetic `user-123` and `role-g`. This model combined with validation data from Morpheus examples can be used to test the DFP Morpheus pipeline. It has little utility outside of testing.
### Input
aws-cloudtrail logs
### Output
Anomalous score of Autoencoder, Binary classification of time series anomaly detection
### References
<!-- Vale attempts to spell check the author names -->
<!-- vale off -->
- https://github.com/AlliedToasters/dfencoder/blob/master/dfencoder/autoencoder.py
- https://github.com/rapidsai/clx/blob/branch-22.12/notebooks/anomaly_detection/FFT_Outlier_Detection.ipynb
- Rasheed Peng Alhajj Rokne Jon: Fourier Transform Based Spatial Outlier Mining 2009 - https://link.springer.com/chapter/10.1007/978-3-642-04394-9_39
<!-- vale on -->

## Flexible Log Parsing
### Model Overview
This model is an example of using Named Entity Recognition (NER) for log parsing, specifically apache web logs.
### Model Architecture
BERT-based cased transformer model with NER classification layer
### Training
Training consisted of fine-tuning the original pretrained [model from Google](https://huggingface.co/bert-base-cased). The labeled training dataset is 1000 parsed apache web logs from a public dataset [Loghub](https://github.com/logpai/loghub)
### How To Use This Model
This model is one example of a BERT-model trained to parse raw logs. It can be used to parse apache web logs or retrained to parse other types of logs as well. The model file has a corresponding config.json file with the names of the fields it parses.
#### Input
raw apache web logs
#### Output
parsed apache web log as jsonlines
### References
- Devlin J. et al. (2018), BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
- https://arxiv.org/abs/1810.04805
- https://medium.com/rapids-ai/cybert-28b35a4c81c4
- https://www.splunk.com/en_us/blog/it/how-splunk-is-parsing-machine-logs-with-machine-learning-on-nvidia-s-triton-and-morpheus.html

## Fraud Detection
### Model Overview
This model shows an application of a graph neural network for fraud detection in a credit card transaction graph. A transaction dataset that includes three types of nodes, transaction, client, and merchant nodes is used for modeling. A combination of `GraphSAGE` along `XGBoost` is used to identify frauds in the transaction networks.
### Model Architecture
It uses a bipartite heterogeneous graph representation as input for `GraphSAGE` for feature learning and `XGBoost` as a classifier. Since the input graph is heterogeneous, a heterogeneous implementation of `GraphSAGE` (HinSAGE) is used for feature embedding.
### Training
A training data consists of raw 753 labeled credit card transaction data with data augmentation in a total of 12053 labeled transaction data. The `GraphSAGE` is trained to output embedded representation of transactions out of the graph. The `XGBoost` is trained using the embedded features as a binary classifier to classify fraud and genuine transactions.
### How To Use This Model
This model is an example of a fraud detection pipeline using a graph neural network and gradient boosting trees. This can be further retrained or fine-tuned to be used for similar types of transaction networks with similar graph structures.
#### Input
Transaction data with nodes including transaction, client, and merchant.
#### Output
An anomalous score of transactions indicates a probability score of being a fraud.
### References
<!-- Vale attempts to spell check the author names -->
<!-- vale off -->
- https://stellargraph.readthedocs.io/en/stable/hinsage.html?highlight=hinsage
- https://github.com/rapidsai/clx/blob/branch-22.12/examples/forest_inference/xgboost_training.ipynb
- Rafaël Van Belle, Charles Van Damme, Hendrik Tytgat, Jochen De Weerdt,Inductive Graph Representation Learning for fraud detection (https://www.sciencedirect.com/science/article/abs/pii/S0957417421017449)
<!-- vale on -->

## Ransomware Detection via AppShield
### Model Overview
This model shows an application of DOCA AppShield to use data from volatile memory to classify processes as ransomware or benign. This model uses a sliding window over time and feeds derived data into a random forest classifiers of various lengths depending on the amount of data collected.
### Model Architecture
The model uses input from Volatility plugins in DOCA AppShield to aggregate and derive features over snapshots in time. The features are used as input into three random forest binary classifiers.
### Training
Training data consists of 87968 labeled AppShield processes from 32 snapshots collected from 256 unique benign and ransomware activities.
### How To Use This Model
Combined with host data from DOCA AppShield, this model can be used to detect ransomware. A training notebook is also included so that users can update the model as more labeled data is collected.
#### Input
Snapshots collected from DOCA AppShield
#### Output
For each `process_id` and snapshot there is a probability score between 1 and 0, where 1 is ransomware and 0 is benign.
### References
<!-- Vale attempts to spell check the author names -->
<!-- vale off -->
- Cohen, A,. & Nissim, N. (2018). Trusted detection of ransomware in a private cloud using machine learning methods leveraging meta-features from volatile memory. In Expert Systems With Applications. (https://www.sciencedirect.com/science/article/abs/pii/S0957417418301283)
- https://developer.nvidia.com/networking/doca
<!-- vale on -->

## Root Cause Analysis
### Model Overview
Root cause analysis is a binary classifier differentiating between ordinary logs and errors/problems/root causes in the log files.
### Model Architecture
BERT-base uncased transformer model
### Training
Training consisted of fine-tuning the original pre-trained [model from google](https://huggingface.co/bert-base-uncased). The labeled dataset is Linux kernel logs, and it has two parts. Kernel errors and new errors. Kernel logs will be split into two parts so that the new and unseen error logs can be appended to the test set after the split to later check if the model can catch them despite not seeing such errors in the training.
### How To Use This Model
This model is an example of customized transformer-based root cause analysis. It can be further fine-tuned for specific root cause analysis or predictive maintenance needs and of your enterprise using the fine-tuning scripts in the repo. The hyper parameters can be optimised to adjust to get the best results with your dataset. The aim is to get the model to predict some false positives that could be previously unknown error types. Users can use this root cause analysis method with other log types too. If they have known failures in their logs, they can use them to train along with ordinary logs and can detect other root causes they weren't aware of before.
#### Input
Kernel logs
#### Output
Binary sequence classification as ordinary or root cause
### References
- Devlin J. et al. (2018), BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
https://arxiv.org/abs/1810.04805
