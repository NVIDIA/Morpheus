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



# Modules

## Core Modules

```{toctree}
:maxdepth: 20

./core/data_loader.md
./core/file_batcher.md
./core/file_to_df.md
./core/filter_control_message.md
./core/filter_detections.md
./core/from_control_message.md
./core/mlflow_model_writer.md
./core/payload_batcher.md
./core/serialize.md
./core/to_control_message.md
./core/write_to_elasticsearch.md
./core/write_to_file.md
```

## Examples Modules

### Digital Fingerprinting Modules

```{toctree}
:maxdepth: 20

./examples/digital_fingerprinting/dfp_data_prep.md
./examples/digital_fingerprinting/dfp_deployment.md
./examples/digital_fingerprinting/dfp_inference_pipe.md
./examples/digital_fingerprinting/dfp_inference.md
./examples/digital_fingerprinting/dfp_monitor.md
./examples/digital_fingerprinting/dfp_postprocessing.md
./examples/digital_fingerprinting/dfp_preproc.md
./examples/digital_fingerprinting/dfp_rolling_window.md
./examples/digital_fingerprinting/dfp_split_users.md
./examples/digital_fingerprinting/dfp_training_pipe.md
./examples/digital_fingerprinting/dfp_training.md

```

### Spear Phishing Modules

```{toctree}
:maxdepth: 20

./examples/spear_phishing/sp_spear_phishing_pre_inference.md
./examples/spear_phishing/sp_spear_phishing_post_inference.md

./examples/spear_phishing/sp_email_enrichment.md
./examples/spear_phishing/sp_inference_sp_classifier.md
./examples/spear_phishing/sp_inference_intent.md
./examples/spear_phishing/sp_label_and_score.md
./examples/spear_phishing/sp_preprocessing.md
./examples/spear_phishing/sp_sender_sketch_aggregator.md
./examples/spear_phishing/sp_sender_sketch_query_constructor.md
./examples/spear_phishing/sp_sender_sketch_update.md


```
