<!--
SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# GNN Fraud Detection Pipeline

## Supported Environments
All environments require additional Conda packages which can be installed with either the `conda/environments/all_cuda-125_arch-$(arch).yaml` or `conda/environments/examples_cuda-125_arch-$(arch).yaml` environment files. Refer to the [Requirements](#requirements) section for more information.
| Environment | Supported | Notes |
|-------------|-----------|-------|
| Conda | ✔ | |
| Morpheus Docker Container | ✔ |  |
| Morpheus Release Container | ✔ |  |
| Dev Container | ✔ |  |

### Supported Architectures
| Architecture | Supported | Issue |
|--------------|-----------|-------|
| x86_64 | ✔ | |
| aarch64 | ✘ | [#2123](https://github.com/nv-morpheus/Morpheus/issues/2123) |

## Requirements

Prior to running the GNN fraud detection pipeline, additional requirements must be installed in to your Conda environment.

```bash
conda env update --solver=libmamba \
  -n ${CONDA_DEFAULT_ENV} \
  --file ./conda/environments/examples_cuda-125_arch-$(arch).yaml
```

## Running
Use Morpheus to run the GNN fraud detection Pipeline with the transaction data. A pipeline has been configured in `run.py` with several command line options:

```bash
python examples/gnn_fraud_detection_pipeline/run.py --help
```
```
Usage: run.py [OPTIONS]

Options:
  --num_threads INTEGER RANGE     Number of internal pipeline threads to use.
                                  [x>=1]
  --pipeline_batch_size INTEGER RANGE
                                  Internal batch size for the pipeline. Can be
                                  much larger than the model batch size. Also
                                  used for Kafka consumers.  [x>=1]
  --model_max_batch_size INTEGER RANGE
                                  Max batch size to use for the model.  [x>=1]
  --model_fea_length INTEGER RANGE
                                  Features length to use for the model.
                                  [x>=1]
  --input_file PATH               Input data filepath.  [required]
  --training_file PATH            Training data filepath.  [required]
  --model_dir PATH                Trained model directory path  [required]
  --output_file TEXT              The path to the file where the inference
                                  output will be saved.
  --help                          Show this message and exit.
```

To launch the configured Morpheus pipeline, run the following:

```bash
python examples/gnn_fraud_detection_pipeline/run.py
```
```
====Registering Pipeline====
====Building Pipeline====
====Building Pipeline Complete!====
====Registering Pipeline Complete!====
====Starting Pipeline====
====Pipeline Started====
====Building Segment: linear_segment_0====
Added source: <from-file-0; FileSourceStage(filename=/examples/gnn_fraud_detection_pipeline/validation.csv, iterative=False, file_type=FileTypes.Auto, repeat=1, filter_null=False, filter_null_columns=None, parser_kwargs=None)>
  └─> morpheus.MessageMeta
Added stage: <deserialize-1; DeserializeStage(ensure_sliceable_index=True, task_type=None, task_payload=None)>
  └─ morpheus.MessageMeta -> morpheus.ControlMessage
Added stage: <fraud-graph-construction-2; FraudGraphConstructionStage(training_file=/examples/gnn_fraud_detection_pipeline/training.csv)>
  └─ morpheus.ControlMessage -> morpheus.ControlMessage
Added stage: <monitor-3; MonitorStage(description=Graph construction rate, smoothing=0.05, unit=messages, delayed_start=False, determine_count_fn=None, log_level=LogLevels.INFO)>
  └─ morpheus.ControlMessage -> morpheus.ControlMessage
Added stage: <gnn-fraud-sage-4; GraphSAGEStage(model_dir=/examples/gnn_fraud_detection_pipeline/model, batch_size=100, record_id=index, target_node=transaction)>
  └─ morpheus.ControlMessage -> morpheus.ControlMessage
Added stage: <monitor-5; MonitorStage(description=Inference rate, smoothing=0.05, unit=messages, delayed_start=False, determine_count_fn=None, log_level=LogLevels.INFO)>
  └─ morpheus.ControlMessage -> morpheus.ControlMessage
Added stage: <gnn-fraud-classification-6; ClassificationStage(model_xgb_file=/examples/gnn_fraud_detection_pipeline/model/xgb.pt)>
  └─ morpheus.ControlMessage -> morpheus.ControlMessage
Added stage: <monitor-7; MonitorStage(description=Add classification rate, smoothing=0.05, unit=messages, delayed_start=False, determine_count_fn=None, log_level=LogLevels.INFO)>
  └─ morpheus.ControlMessage -> morpheus.ControlMessage
Added stage: <serialize-8; SerializeStage(include=None, exclude=None, fixed_columns=True)>
  └─ morpheus.ControlMessage -> morpheus.MessageMeta
Added stage: <monitor-9; MonitorStage(description=Serialize rate, smoothing=0.05, unit=messages, delayed_start=False, determine_count_fn=None, log_level=LogLevels.INFO)>
  └─ morpheus.MessageMeta -> morpheus.MessageMeta
Added stage: <to-file-10; WriteToFileStage(filename=.tmp/output/gnn_fraud_detection_output.csv, overwrite=True, file_type=FileTypes.Auto, include_index_col=True, flush=False)>
  └─ morpheus.MessageMeta -> morpheus.MessageMeta
====Building Segment Complete!====
Graph construction rate[Complete]: 265 messages [00:00, 1016.18 messages/s]
Inference rate[Complete]: 265 messages [00:00, 545.08 messages/s]
Add classification rate[Complete]: 265 messages [00:00, 492.11 messages/s]
Serialize rate[Complete]: 265 messages [00:00, 480.77 messages/s]
====Pipeline Complete====
```

### CLI Example
The above example is illustrative of using the Python API to build a custom Morpheus pipeline. Alternately, the Morpheus command line could have been used to accomplish the same goal. To do this we must ensure the `examples` directory is available in the `PYTHONPATH` and each of the custom stages are registered as plugins.

Note: Since the `gnn_fraud_detection_pipeline` module is visible to Python we can specify the plugins by their module name rather than the more verbose file path.

From the root of the Morpheus repo, run:
```bash
PYTHONPATH="examples" \
morpheus --log_level INFO \
	--plugin "gnn_fraud_detection_pipeline" \
	run --pipeline_batch_size 1024 --model_max_batch_size 32 --edge_buffer_size 4 \
	pipeline-other --model_fea_length 70 --label=probs \
	from-file --filename examples/gnn_fraud_detection_pipeline/validation.csv --filter_null False \
	deserialize \
	fraud-graph-construction --training_file examples/gnn_fraud_detection_pipeline/training.csv \
	monitor --description "Graph construction rate" \
	gnn-fraud-sage --model_dir  examples/gnn_fraud_detection_pipeline/model/ \
	monitor --description "Inference rate" \
	gnn-fraud-classification --model_xgb_file examples/gnn_fraud_detection_pipeline/model/xgb.pt \
	monitor --description "Add classification rate" \
	serialize \
	to-file --filename ".tmp/output/gnn_fraud_detection_cli_output.csv" --overwrite
```
