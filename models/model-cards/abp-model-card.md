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


# Anomalous Behavior Profiling (ABP)

# Model Overview

## Description:
* This model is an example of a binary XGBoost classifier to differentiate between anomalous GPU behavior, such as cryptocurrency mining / GPU malware, and non-anomalous GPU-based workflows (for example, ML/DL training). This model is for demonstration purposes and not for production usage. <br>

## References:
<!-- Vale attempts to spell check the author names -->
<!-- vale off -->
* Chen, Guestrin (2016) XGBoost. A scalable tree boosting system. https://arxiv.org/abs/1603.02754  <br>
<!-- vale on -->

## Model Architecture:
**Architecture Type:**
* Gradient boosting <br>

**Network Architecture:**
* XGBoost <br>

## Input: (Enter "None" As Needed)

**Input Format:**
* `nvidia-smi` output <br>

**Input Parameters:**
* GPU statistics that are included in the `nvidia-smi` output <br>

**Other Properties Related to Output:** N/A <br>

## Output: (Enter "None" As Needed)

**Output Format:**
* Binary Results <br>

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
* https://github.com/nv-morpheus/Morpheus/blob/branch-25.02/models/datasets/training-data/abp-sample-nvsmi-training-data.json  <br>

**Properties (Quantity, Dataset Descriptions, Sensors):**
* Sample dataset consists of over 1000 `nvidia-smi` outputs <br>

## Evaluation Dataset:

**Link:**
* https://github.com/nv-morpheus/Morpheus/blob/branch-25.02/models/datasets/validation-data/abp-validation-data.jsonlines  <br>

**Properties (Quantity, Dataset Descriptions, Sensors):**
* Sample dataset consists of over 1000 `nvidia-smi` outputs <br>

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
* The model is primarily designed for testing purposes and serves as a small model specifically used to evaluate and validate the ABP pipeline. Its application is focused on assessing the effectiveness of the pipeline rather than being intended for broader use cases or specific applications beyond testing.

### Intended Users.
* The model is primarily designed for testing purposes. This model is intended to be an example for developers that want to test Morpheus ABP pipeline.

### Name who is intended to benefit from this model.
* The intended beneficiaries of this model are developers who aim to test the functionality of the ABP models for detecting cryptocurrency mining.

### Describe the model output.
* This model output can be used as a binary result, cryptocurrency mining or legitimate GPU usage.

### Describe how this model works.
* `nvidia-smi` features are used as the input and the model predicts a label for each row

### List the technical limitations of the model.
* For different GPU workloads different models need to be trained.

### Has this been verified to have met prescribed NVIDIA quality standards?
* Yes

### What performance metrics were used to affirm the model's performance?
* Accuracy

### What are the potential known risks to users and stakeholders?
* N/A

### Link the relevant end user license agreement
* [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0)


## Model Card ++ Safety & Security Subcard

### Link the location of the repository for the training dataset.
* https://github.com/nv-morpheus/Morpheus/blob/branch-25.02/models/datasets/training-data/abp-sample-nvsmi-training-data.json

### Describe the life critical impact (if present).
* None

### Was model and dataset assessed for vulnerability for potential form of attack?
* No

### Name applications for the model.
* The primary application for this model is testing the Morpheus pipeline.

### Name use case restrictions for the model.
* The model's use case is restricted to testing the Morpheus pipeline and may not be suitable for other applications.

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
* Yes

### Does data labeling (annotation, metadata) comply with privacy laws?
* N/A

### Is data compliant with data subject requests for data correction or removal, if such a request was made?
* N/A
