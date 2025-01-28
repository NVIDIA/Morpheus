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


# Root Cause Analysis

# Model Overview

## Description:
* Root cause analysis is a binary classifier differentiating between ordinary logs and errors/problems/root causes in the log files. <br>

## References:
* Devlin J. et al. (2018), BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding https://arxiv.org/abs/1810.04805 <br>

## Model Architecture:
**Architecture Type:**
* Transformers <br>

**Network Architecture:**
* BERT <br>

## Input: (Enter "None" As Needed)

**Input Format:**
* CSV <br>

**Input Parameters:**
* kern.log file contents <br>

**Other Properties Related to Output:**
* N/A <br>

## Output: (Enter "None" As Needed)

**Output Format:**
* Binary Results, Root Cause or Ordinary <br>

**Output Parameters:**
* N/A <br>

**Other Properties Related to Output:**
* N/A <br>

## Software Integration:

**Runtime:**
* Morpheus  <br>

**Supported Hardware Platforms:** <br>
* Ampere/Turing <br>

**Supported Operating Systems:** <br>
* Linux <br>

## Model Versions:
* v1  <br>

# Training & Evaluation:

## Training Dataset:

**Link:**
* https://github.com/nv-morpheus/Morpheus/blob/branch-25.02/models/datasets/training-data/root-cause-training-data.csv <br>

**Properties (Quantity, Dataset Descriptions, Sensors):**
* kern.log files from DGX machines <br>

## Evaluation Dataset:

**Link:**
* https://github.com/nv-morpheus/Morpheus/blob/branch-25.02/models/datasets/validation-data/root-cause-validation-data-input.jsonlines  <br>

**Properties (Quantity, Dataset Descriptions, Sensors):**
* kern.log files from DGX machines <br>

## Inference:

**Engine:**
* Triton <br>

**Test Hardware:** <br>
* Other  <br>

# Subcards

## Model Card ++ Bias Subcard

### Individuals from the following adversely impacted (protected classes) groups participate in model design and testing.
* Not Applicable

### Describe measures taken to mitigate against unwanted bias.
* Not Applicable

## Model Card ++ Explainability Subcard

### Name example applications and use cases for this model.
* The model is primarily designed for testing purposes and serves as a small pre-trained model specifically used to evaluate and validate the Root Cause Analysis pipeline. This model is an example of customized transformer-based root cause analysis. It can be used for pipeline testing purposes. It needs to be re-trained for specific root cause analysis or predictive maintenance needs with the fine-tuning scripts in the repo. The hyperparameters can be optimised to adjust to get the best results with another dataset. The aim is to get the model to predict some false positives that could be previously unknown error types. Users can use this root cause analysis approach with other log types too. If they have known failures in their logs, they can use them to train along with ordinary logs and can detect other root causes they weren't aware of before.

### Intended Users.
* This model is designed for developers seeking to test the root cause analysis pipeline with a small pre-trained model trained on a very small `kern.log` file from a DGX.

### Name who is intended to benefit from this model.
* The intended beneficiaries of this model are developers who aim to test the functionality of the DFP pipeline using synthetic datasets

### Describe the model output.
* This model output can be used as a binary result, Root cause or Ordinary

### Describe how this model works.
* A BERT model gets fine-tuned with the kern.log dataset and in the inference it predicts one of the binary classes. Root cause or Ordinary.

### List the technical limitations of the model.
* For different log types and content, different models need to be trained.

### Has this been verified to have met prescribed NVIDIA quality standards?
* Yes

### What performance metrics were used to affirm the model's performance?
* F1

### What are the potential known risks to users and stakeholders?
* N/A

### Link the relevant end user license agreement
* [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0)<br>


## Model Card ++ Safety & Security Subcard

### Link the location of the repository for the training dataset.
* https://github.com/nv-morpheus/Morpheus/blob/branch-25.02/models/datasets/training-data/root-cause-training-data.csv

### Describe the life critical impact (if present).
* None

### Was model and dataset assessed for vulnerability for potential form of attack?
* No

### Name applications for the model.
* The primary application for this model is testing the Morpheus pipeline.

### Name use case restrictions for the model.
* Different models need to be trained depending on the log types.

### Is the model and dataset compliant with National Classification Management Society (NCMS)?
* No

### Describe access restrictions

* The Principle of least privilege (PoLP) is applied limiting access for dataset generation and model development. Restrictions enforce dataset access during training, and dataset license constraints adhered to.

### Is there a digital signature?
* No

## Model Card ++ Privacy Subcard


### Generatable or reverse engineerable personally-identifiable information (PII)?
* None

### Was consent obtained for any PII used?
* N/A

### Protected classes used to create this model? (The following were used in model the model's training:)
* N/A

### How often is dataset reviewed?
* The dataset is initially reviewed upon addition, and subsequent reviews are conducted as needed or upon request for any changes.

### Is a mechanism in place to honor data subject right of access or deletion of personal data?
* N/A

### If PII collected for the development of this AI model, was it minimized to only what was required?
* N/A

### Is there data provenance?
* Original raw logs are not saved. The small sample in the repo is saved for testing the pipeline.

### Does data labeling (annotation, metadata) comply with privacy laws?
* N/A

### Is data compliant with data subject requests for data correction or removal, if such a request was made?
* N/A
