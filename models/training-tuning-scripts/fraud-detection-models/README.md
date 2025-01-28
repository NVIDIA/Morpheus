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



## Instruction how to train new GNN models.

### Setup training environment

Install packages for training GNN model.

```bash
mamba env update \
  -n ${CONDA_DEFAULT_ENV} \
  --file ./conda/environments/model-utils-125_arch-$(arch).yaml
```

### Options for training and tuning models.

```bash
python training.py --help
Usage: training.py [OPTIONS]

Options:
  --training-data TEXT    Path to training data
  --validation-data TEXT  Path to validation data
  --model-dir TEXT        path to model directory
  --target-node TEXT      Target node
  --epochs INTEGER        Number of epochs
  --batch_size INTEGER    Batch size
  --output-file TEXT      Path to csv inference result
  --model-type TEXT       Model type either RGCN/HinSAGE
  --help                  Show this message and exit

```


#### Example usage:

```bash
export DATASET=../../dataset

python training.py --training-data $DATASET/training-data/fraud-detection-training-data.csv \
--validation-data $DATASET\validation-datafraud-detection-validation-data.csv \
         --epochs 20 \
         --model_dir models\
         --model-type HinSAGE
```
This results is a trained models of HeteroRGCN/HinSAGE (model.pt) and Gradient boosting tree (xgb.pt), hyperparameters at the `model` directory.

Note the `model.py` used for both training & inference script is a symbolic link for the `../../../examples/gnn_fraud_detection_pipeline/stages/model.py`.
