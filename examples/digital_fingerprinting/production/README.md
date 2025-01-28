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

# "Production" Digital Fingerprinting Pipeline

This example is designed to illustrate a full-scale, production-ready, DFP deployment in Morpheus. It contains all of the necessary components (such as a model store), to allow multiple Morpheus pipelines to communicate at a scale that can handle the workload of an entire company.

Key Features:
 * Multiple pipelines are specialized to perform either training or inference
 * Uses a model store to allow the training and inference pipelines to communicate
 * Organized into a `docker compose` deployment for easy startup
 * Contains a Jupyter notebook service to ease development and debugging
 * Can be deployed to Kubernetes using provided Helm charts
 * Uses many customized stages to maximize performance.

## Building and Running via `docker compose`
### Build
```bash
cd examples/digital_fingerprinting/production
docker compose build
```

> **Note:** This requires version 1.28.0 or higher of Docker Compose, and preferably v2. If you encounter an error similar to:
>
> ```
> ERROR: The Compose file './docker-compose.yml' is invalid because:
> services.jupyter.deploy.resources.reservations value Additional properties are not allowed ('devices' was
> unexpected)
> ```
>
> This is most likely due to using an older version of the `docker-compose` command, instead re-run the build with `docker compose`. Refer to [Migrate to Compose V2](https://docs.docker.com/compose/migrate/) for more information.

### Fetch Example Data
The `examples/digital_fingerprinting/fetch_example_data.py` script can be used to fetch the Duo and Azure logs to run the example pipelines.

Download the data needed to run the DFP pipeline on Azure / Duo logs:
```bash
docker compose run fetch_data
```


### Running the services

The Morpheus DFP pipeline can be run from either a Jupyter Notebook using the `jupyter` service or from the command line using the `morpheus_pipeline` service. The `mlflow` service is also started in the background to provide a tracking URI for the Morpheus pipeline.

#### Optional MLflow Service
Starting either the `morpheus_pipeline` or the `jupyter` service, will start the `mlflow` service in the background. For debugging purposes it can be helpful to view the logs of the running MLflow service.

From the `examples/digital_fingerprinting/production` dir run:
```bash
docker compose up mlflow
```

By default, a MLflow dashboard will be available at:
```bash
http://localhost:5000
```

#### Jupyter Server
From the `examples/digital_fingerprinting/production` dir run:
```bash
docker compose up jupyter
```

Once the build is complete and the service has started, a message similar to the following should display:
```
jupyter  |     To access the server, open this file in a browser:
jupyter  |         file:///root/.local/share/jupyter/runtime/jpserver-7-open.html
jupyter  |     Or copy and paste one of these URLs:
jupyter  |         http://localhost:8888/lab?token=<token>
jupyter  |      or http://127.0.0.1:8888/lab?token=<token>
```

Copy and paste the URL into a web browser. There are six notebooks included with the DFP example:
* dfp_azure_inference.ipynb - Inference pipeline for Azure Active Directory data
* dfp_azure_integrated_training.ipynb - Integrated training pipeline for Azure Active Directory data
* dfp_azure_training.ipynb - Training pipeline for Azure Active Directory data
* dfp_duo_inference.ipynb - Inference pipeline for Duo Authentication
* dfp_duo_integrated_training.ipynb - Integrated training pipeline for Duo Authentication
* dfp_duo_training.ipynb - Training pipeline for Duo Authentication

> **Note:** The token in the URL is a one-time use token, and a new one is generated with each invocation.

#### Morpheus Pipeline Service
By default the `morpheus_pipeline` will run the training pipeline for Duo data, from the `examples/digital_fingerprinting/production` dir run:
```bash
docker compose up morpheus_pipeline
```

If instead you want to run a different pipeline, from the `examples/digital_fingerprinting/production` dir run:
```bash
docker compose run morpheus_pipeline bash
```

From the prompt within the `morpheus_pipeline` container you can run either the `dfp_azure_pipeline.py` or `dfp_duo_pipeline.py` pipeline scripts.
```bash
python dfp_azure_pipeline.py --help
python dfp_duo_pipeline.py --help
```

Both scripts are capable of running either a training or inference pipeline for their respective data sources. The command line options for both are the same:
| Flag | Type | Description |
| ---- | ---- | ----------- |
| `--train_users` | One of: `all`, `generic`, `individual`, `none` | Indicates whether or not to train per user or a generic model for all users. Selecting `none` runs the inference pipeline. |
| `--skip_user` | TEXT | User IDs to skip. Mutually exclusive with `only_user` |
| `--only_user` | TEXT | Only users specified by this option will be included. Mutually exclusive with `skip_user` |
| `--start_time` | TEXT | The start of the time window, if undefined `start_date` will be `now()-duration` |
| `--duration` | TEXT | The duration to run starting from now [default: 60d] |
| `--cache_dir` | TEXT | The location to cache data such as S3 downloads and pre-processed data  [environment variable: `DFP_CACHE_DIR`; default: `./.cache/dfp`] |
| `--log_level` | One of: `CRITICAL`, `FATAL`, `ERROR`, `WARN`, `WARNING`, `INFO`, `DEBUG` | Specify the logging level to use.  [default: `WARNING`] |
| `--sample_rate_s` | INTEGER | Minimum time step, in milliseconds, between object logs.  [environment variable: `DFP_SAMPLE_RATE_S`; default: 0] |
| `-f`, `--input_file` | TEXT | List of files to process. Can specify multiple arguments for multiple files. Also accepts glob (*) wildcards and schema prefixes such as `s3://`. For example, to make a local cache of an s3 bucket, use `filecache::s3://mybucket/*`. Refer to [`fsspec` documentation](https://filesystem-spec.readthedocs.io/en/latest/api.html?highlight=open_files#fsspec.open_files) for list of possible options. |
| `--watch_inputs` | FLAG | Instructs the pipeline to continuously check the paths specified by `--input_file` for new files. This assumes that the at least one paths contains a wildcard. |
| `--watch_interval` | FLOAT | Amount of time, in seconds, to wait between checks for new files. Only used if --watch_inputs is set. [default `1.0`] |
| `--tracking_uri` | TEXT | The MLflow tracking URI to connect to. [default: `http://localhost:5000`] |
| `--help` | | Show this message and exit. |

##### Steps to Run Example Pipeline

Run Duo Training Pipeline:
```bash
python dfp_duo_pipeline.py --train_users generic --start_time "2022-08-01" --input_file="../../data/dfp/duo-training-data/*.json"
```

Run Duo Inference Pipeline:
```bash
python dfp_duo_pipeline.py --train_users none --start_time "2022-08-30" --input_file="../../data/dfp/duo-inference-data/*.json"
```

Run Azure Training Pipeline:

```bash
python dfp_azure_pipeline.py --train_users generic --start_time "2022-08-01" --input_file="../../data/dfp/azure-training-data/AZUREAD_2022*.json"
```

Run Azure Inference Pipeline:
```bash
python dfp_azure_pipeline.py --train_users none  --start_time "2022-08-30" --input_file="../../data/dfp/azure-inference-data/*.json"
```

## Additional Information

Additional information on the DFP pipeline, including how it can be customized for additional data sources can be found in the [Digital Fingerprinting Guide](../../../docs/source/developer_guide/guides/5_digital_fingerprinting.md) as well as the [Digital Fingerprinting Reference Guide](../../../docs/source/developer_guide/guides/6_digital_fingerprinting_reference.md).

The commands in the previous section run stage-based example DFP pipelines. The Morpheus 23.03 release introduced a new, more flexible module-based approach to build pipelines through the use of control messages. More information about modular DFP pipelines can be found at [Introduction to Modular Digital Fingerprinting Pipeline Guide](../../../docs/source/developer_guide/guides/10_modular_pipeline_digital_fingerprinting.md). Along with the commands to run equivalent module-based DFP pipelines are available at [Running Example Modular DFP Pipelines](../../../docs/source/developer_guide/guides/10_modular_pipeline_digital_fingerprinting.md#running-example-modular-dfp-pipelines).
