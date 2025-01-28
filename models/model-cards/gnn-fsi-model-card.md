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

# Model Overview

### Description:
* This model shows an application of a graph neural network for fraud detection in a credit card transaction graph. A transaction dataset that includes three types of nodes, transaction, client, and merchant nodes is used for modeling. A combination of `GraphSAGE` along `XGBoost` is used to identify frauds in the transaction networks. This model is for demonstration purposes and not for production usage. <br>

## References:
<!-- Vale attempts to spell check the author names -->
<!-- vale off -->
1. https://stellargraph.readthedocs.io/en/stable/hinsage.html?highlight=hinsage
2. https://github.com/rapidsai/clx/blob/branch-22.12/examples/forest_inference/xgboost_training.ipynb
3. Rafaël Van Belle, Charles Van Damme, Hendrik Tytgat, Jochen De Weerdt, Inductive Graph Representation Learning for fraud detection (https://www.sciencedirect.com/science/article/abs/pii/S0957417421017449)<br>
<!-- vale o -->

## Model Architecture:
It uses a bipartite heterogeneous graph representation as input for `GraphSAGE` for feature learning and `XGBoost` as a classifier. Since the input graph is heterogeneous, a heterogeneous implementation of `GraphSAGE` (HinSAGE) is used for feature embedding.<br>
**Architecture Type:**
* Graph Neural Network and Binary classification <br>

**Network Architecture:**
* GraphSAGE and XGBoost <br>

## Input
Transaction data with nodes including transaction, client, and merchant.<br>
**Input Parameters:**
* None <br>

**Input Format:**
* CSV format<br>

**Other Properties Related to Output:**
* None<br>

## Output
An anomalous score of transactions indicates a probability score of being a fraud.<br>
**Output Parameters:**
* None <br>

**Output Format:**
* CSV<br>

**Other Properties Related to Output:**
* None <br>

## Software Integration:
**Runtime:**
* Morpheus  <br>

**Supported Hardware Platforms:** <br>
* Ampere/Turing <br>

**Supported Operating Systems:** <br>
* Linux <br>

## Model Versions:
* 1.0 <br>

### How To Use This Model
This model is an example of a fraud detection pipeline using a graph neural network and gradient boosting trees. This can be further retrained or fine-tuned to be used for similar types of transaction networks with similar graph structures.

# Training & Evaluation:

## Training Dataset:

**Link:**
* [fraud-detection-training-data.csv](models/dataset/fraud-detection-training-data.csv)  <br>

**Properties (Quantity, Dataset Descriptions, Sensors):**
* A training data consists of raw 753 synthetic labeled credit card transaction data with data augmentation in a total of 12053 labeled transaction data. <br>

## Evaluation Dataset:
**Link:**
* [fraud-detection-validation-data.csv](models/dataset/fraud-detection-validation-data.csv)  <br>

**Properties (Quantity, Dataset Descriptions, Sensors):**
* Data consists of raw 265 labeled credit card transaction synthetically created<br>

## Inference:
**Engine:**
* Triton <br>

**Test Hardware:** <br>
* DGX (V100) <br>

## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their supporting model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse. For more detailed information on ethical considerations for this model, please see the Model Card++ Explainability, Bias, Safety & Security, and Privacy Subcards below. Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).

# Subcards
## Model Card ++ Bias Subcard

### Individuals from the following adversely impacted (protected classes) groups participate in model design and testing.
* Not Applicable

### Describe measures taken to mitigate against unwanted bias.
* Not Applicable

## Model Card ++ Explainability Subcard

### Name example applications and use cases for this model.
* The model is primarily designed for testing purposes and serves as a small pretrained model specifically used to evaluate and validate the GNN FSI pipeline. Its application is focused on assessing the effectiveness of the pipeline rather than being intended for broader use cases or specific applications beyond testing.

### Fill in the blank for the model technique.
* This model is designed for developers seeking to test the GNN fraud detection pipeline with a small pretrained model on a synthetic dataset.

### Intended Users.
* The intended beneficiaries of this model are developers who aim to test the performance and functionality of the GNN fraud detection pipeline using synthetic datasets. It may not be suitable or provide significant value for real-world transactions.

### Describe the model output.
* This model outputs fraud probability score b/n (0 & 1).

### Describe how this model works.
* The model uses a bipartite heterogeneous graph representation as input for `GraphSAGE` for feature learning and `XGBoost` as a classifier. Since the input graph is heterogeneous, a heterogeneous implementation of `GraphSAGE` (HinSAGE) is used for feature embedding.

### List the technical limitations of the model.
* This model version requires a transactional data schema with entities (user, merchant, transaction) as requirement for the model.

### Has this been verified to have met prescribed NVIDIA standards?
* Yes

### What performance metrics were used to affirm the model's performance?
* Area under ROC curve and Accuracy

### What are the potential known risks to users and stakeholders?
* None

### Link the relevant end user license agreement
* [Apache 2.0](https://github.com/nv-morpheus/Morpheus/blob/branch-25.02/LICENSE)

## Model Card ++ Safety & Security Subcard

### Link the location of the repository for the training dataset (if able to share).
* [training dataset](models/datasets/training-data/fraud-detection-training-data.csv)

### Describe the life critical impact (if present).
* None

### Was model and dataset assessed for vulnerability for potential form of attack?
* No

### Name applications for the model.
* Used for testing fraud detection application in Morpheus pipeline, under the defined dataset schema description.

### Name use case restrictions for the model.
* The model's use case is restricted to testing the Morpheus pipeline and may not be suitable for other applications.

### Is the model and dataset compliant with National Classification Management Society (NCMS)?
* Not Applicable

### Describe access restrictions

* The Principle of least privilege (PoLP) is applied limiting access for dataset generation and model development. Restrictions enforce dataset access during training, and dataset license constraints adhered to.

### Is there a digital signature?
* No

## Model Card ++ Privacy Subcard

### Generatable or reverse engineerable personally-identifiable information (PII)?
* None

### Protected classes used to create this model? (The following were used in model the model's training:)
* Not applicable

### Was consent obtained for any PII used?
* Not Applicable (Data is extracted from synthetically created credit card transaction,refer[3] for the source of data creation)

### How often is dataset reviewed?
* The dataset is initially reviewed upon addition, and subsequent reviews are conducted as needed or upon request for any changes.

### Is a mechanism in place to honor data subject right of access or deletion of personal data?
* Yes

### If PII collected for the development of this AI model, was it minimized to only what was required?
* Not applicable

### Is there data provenance?
* Yes

### Does data labeling (annotation, metadata) comply with privacy laws?
* Not applicable

### Is data compliant with data subject requests for data correction or removal, if such a request was made?
* Not applicable
