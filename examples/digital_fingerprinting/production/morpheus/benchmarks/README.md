<!--
# Copyright (c) 2021-2025, NVIDIA CORPORATION.
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
-->

# Running DFP E2E Benchmarks

### Set up Morpheus Dev Container

If you don't already have the Morpheus Dev container, run the following to build it:
```bash
./docker/build_container_dev.sh
```

Now run the container:
```bash
./docker/run_container_dev.sh
```

Note that Morpheus containers are tagged by date. By default, `run_container_dev.sh` will try to use current date as tag. Therefore, if you are trying to run a container that was not built on the current date, you must set the `DOCKER_IMAGE_TAG` environment variable. For example,
```bash
DOCKER_IMAGE_TAG=dev-221003 ./docker/run_container_dev.sh
```

In the `/workspace` directory of the container, run the following to compile Morpheus:
```bash
./scripts/compile.sh
```

Install additional required dependencies:
```bash
conda env update --solver=libmamba \
  -n ${CONDA_DEFAULT_ENV} \
  --file ./conda/environments/examples_cuda-125_arch-$(arch).yaml
```


Fetch input data for benchmarks:
```bash
./examples/digital_fingerprinting/fetch_example_data.py all
```

### Start MLflow

MLflow is used as the model repository where the trained DFP models will be published and used for inference by the pipelines. Run the following to start MLflow in a host terminal window (not container):

```bash
# from root of Morpheus repo
cd examples/digital_fingerprinting/production
```

```bash
docker compose up mlflow
```


### Run E2E Benchmarks

Benchmarks are run using `pytest-benchmark`. By default, there are five rounds of measurement. For each round, there will be one iteration of each workflow. Measurements are taken for each round. Final results such as `min`, `max` and `mean` times will be based on these measurements.

To provide your own calibration or use other `pytest-benchmark` features with these workflows, refer to their [documentation](https://pytest-benchmark.readthedocs.io/en/latest/).

Morpheus pipeline configurations for each workflow are managed using [pipelines_conf.json](./resource/pipelines_conf.json). For example, this is the Morpheus configuration for  `dfp_modules_duo_payload_inference`:
```
"test_dfp_modules_duo_payload_inference_e2e": {
		"message_path": "../control_messages/duo_payload_inference.json",
		"num_threads": 12,
		"pipeline_batch_size": 256,
		"edge_buffer_size": 128,
		"start_time": "2022-08-01",
		"duration": "60d",
		"userid_column_name": "username",
		"timestamp_column_name": "timestamp",
		"source": "duo"
},
...
```

When using the MRC SegmentModule in a pipeline, it will also require a module configuration which gets generated within the test. Additional information is included in the [Morpheus Pipeline with Modules](../../../../../docs/source/developer_guide/guides/6_digital_fingerprinting_reference.md#morpheus-pipeline-with-modules)

To ensure the [file_to_df_loader.py](../../../../../morpheus/loaders/file_to_df_loader.py) utilizes the same type of downloading mechanism, set `MORPHEUS_FILE_DOWNLOAD_TYPE` environment variable with any one of given choices (`dask`, `dask thread`, `single thread`).

```bash
export MORPHEUS_FILE_DOWNLOAD_TYPE=dask
```

Benchmarks for an individual workflow can be run from `examples/digital_fingerprinting/production/morpheus` in your container:

```bash
cd examples/digital_fingerprinting/production/morpheus
pytest -s --log-level=WARN --benchmark-enable --benchmark-warmup=on --benchmark-warmup-iterations=1 --benchmark-autosave benchmarks/test_bench_e2e_dfp_pipeline.py::<test-workflow>
```

The `-s` option allows outputs of pipeline execution to be displayed so you can ensure there are no errors while running your benchmarks.

The `--benchmark-warmup` and `--benchmark-warmup-iterations` options are used to run the workflows once before starting measurements. This is because, if it does not already exist, the preprocessed data is cached during the initial run.

`<test-workflow>` is the name of the test to run benchmarks on. This can be one of the following:
- `test_dfp_modules_azure_payload_inference_e2e`
- `test_dfp_modules_azure_payload_lti_e2e`
- `test_dfp_modules_azure_payload_lti_s3_e2e`
- `test_dfp_modules_azure_payload_training_e2e`
- `test_dfp_modules_azure_streaming_inference_e2e`
- `test_dfp_modules_azure_streaming_lti_e2e`
- `test_dfp_modules_azure_streaming_training_e2e`
- `test_dfp_modules_duo_payload_inference_e2e`
- `test_dfp_modules_duo_payload_lti_e2e`
- `test_dfp_modules_duo_payload_only_load_e2e`
- `test_dfp_modules_duo_payload_training_e2e`
- `test_dfp_modules_duo_streaming_inference_e2e`
- `test_dfp_modules_duo_streaming_lti_e2e`
- `test_dfp_modules_duo_streaming_only_load_e2e`
- `test_dfp_modules_duo_streaming_payload_e2e`
- `test_dfp_modules_duo_streaming_training_e2e`
- `test_dfp_stages_azure_training_e2e`
- `test_dfp_stages_azure_inference_e2e`
- `test_dfp_stages_duo_training_e2e`
- `test_dfp_stages_duo_inference_e2e`

For example, to run E2E benchmarks on the DFP training (modules) workflow on the azure logs:
```bash
pytest -s --benchmark-enable --benchmark-warmup=on --benchmark-warmup-iterations=1 --benchmark-autosave benchmarks/test_bench_e2e_dfp_pipeline.py::test_dfp_modules_azure_payload_lti_e2e
```

To run E2E benchmarks on all workflows:
```bash
pytest -s --benchmark-enable --benchmark-warmup=on --benchmark-warmup-iterations=1 --benchmark-autosave benchmarks/test_bench_e2e_dfp_pipeline.py
```

Here are the benchmark comparisons for individual tests. When the control message type is `payload`, the rolling window stage is bypassed, whereas when it is `streaming`, the windows are created with historical data.

#### Training (Azure):
```
------------------------------------------------------------------------------------------------------------------------------------------
Name (time in s)                                      Min      Max     Mean  StdDev   Median     IQR  Outliers     OPS  Rounds  Iterations
------------------------------------------------------------------------------------------------------------------------------------------
test_dfp_modules_azure_payload_training_e2e        7.5737   8.1568   7.9212  0.2352   7.9819  0.3582       2;0  0.1262       5           1
test_dfp_stages_azure_training_e2e     		  26.4748  28.5647  27.5939  0.8192  27.9162  1.1681       2;0  0.0362       5           1
test_dfp_modules_azure_streaming_training_e2e     27.9206  28.9689  28.3179  0.4313  28.1994  0.6577       1;0  0.0353       5           1
```


#### Inference (Azure):
```
---------------------------------------------------------------------------------------------------------------------------------------
Name (time in s)                                      Min     Max    Mean  StdDev  Median     IQR  Outliers     OPS  Rounds  Iterations
---------------------------------------------------------------------------------------------------------------------------------------
test_dfp_modules_azure_streaming_inference_e2e     1.3149  1.6442  1.4847  0.1179  1.4758  0.1160       2;0  0.6735       5           1
test_dfp_modules_azure_payload_inference_e2e       1.3794  1.7681  1.5854  0.1429  1.6209  0.1684       2;0  0.6307       5           1
test_dfp_stages_azure_inference_e2e     	   1.5218  1.7641  1.6401  0.1096  1.6082  0.2000       2;0  0.6097       5           1
---------------------------------------------------------------------------------------------------------------------------------------
```

#### Training (Duo)
```
----------------------------------------------------------------------------------------------------------------------------------------
Name (time in s)                                    Min      Max     Mean  StdDev   Median     IQR  Outliers     OPS  Rounds  Iterations
----------------------------------------------------------------------------------------------------------------------------------------
test_dfp_modules_duo_payload_training_e2e        5.8277   6.5059   6.2262  0.2614   6.3224  0.3434       2;0  0.1606       5           1
test_dfp_stages_duo_training_e2e     		16.8809  17.6059  17.1800  0.3213  17.0975  0.5700       1;0  0.0582       5           1
test_dfp_modules_duo_streaming_training_e2e     17.3468  18.6124  18.1468  0.5265  18.3208  0.8012       1;0  0.0551       5           1
----------------------------------------------------------------------------------------------------------------------------------------
```

#### Inference (Duo)
```
-------------------------------------------------------------------------------------------------------------------------------------------------
Name (time in ms)                                     Min         Max      Mean   StdDev    Median      IQR  Outliers     OPS  Rounds  Iterations
-------------------------------------------------------------------------------------------------------------------------------------------------
test_dfp_modules_duo_payload_inference_e2e     	 849.7947    974.5900  892.5823  51.2846  886.1437  70.6811       1;0  1.1203       5           1
test_dfp_modules_duo_streaming_inference_e2e     898.7004  1,041.2128  980.8650  57.1454  979.5714  87.5982       2;0  1.0195       5           1
test_dfp_stages_duo_inference_e2e     		 966.6770  1,005.0367  987.8158  17.8514  998.1837  30.8821       1;0  1.0123       5           1
-------------------------------------------------------------------------------------------------------------------------------------------------
```

#### Integrated Training and Inference (Azure)
```
-------------------------------------------------------------------------------------------------------------------------------------
Name (time in s)                                 Min      Max     Mean  StdDev   Median     IQR  Outliers     OPS  Rounds  Iterations
-------------------------------------------------------------------------------------------------------------------------------------
test_dfp_modules_azure_payload_lti_e2e        9.6311  10.4812   9.9689  0.3695   9.8091  0.6150       1;0  0.1003       5           1
test_dfp_modules_azure_streaming_lti_e2e     30.8548  31.5380  31.2297  0.3277  31.3744  0.6081       2;0  0.0320       5           1
-------------------------------------------------------------------------------------------------------------------------------------
```

#### Integrated Training and Inference (Duo)
```
-----------------------------------------------------------------------------------------------------------------------------------
Name (time in s)                               Min      Max     Mean  StdDev   Median     IQR  Outliers     OPS  Rounds  Iterations
-----------------------------------------------------------------------------------------------------------------------------------
test_dfp_modules_duo_payload_lti_e2e        7.5645   8.2497   7.7638  0.2825   7.6664  0.3048       1;0  0.1288       5           1
test_dfp_modules_duo_streaming_lti_e2e     18.9271  20.4393  19.6061  0.5582  19.5907  0.6806       2;0  0.0510       5           1
-----------------------------------------------------------------------------------------------------------------------------------
```

#### Integrated Training and Inference from s3 (Azure)
```
---------------------------------------------------------- benchmark: 1 tests ----------------------------------------------------------
Name (time in s)                                  Min      Max     Mean   StdDev   Median      IQR  Outliers     OPS  Rounds  Iterations
----------------------------------------------------------------------------------------------------------------------------------------
test_dfp_modules_azure_payload_lti_s3_e2e     44.9810  74.9993  59.6432  10.8338  58.9347  12.0293       2;0  0.0168       5           1
----------------------------------------------------------------------------------------------------------------------------------------
```

### Benchmarks Report

Each time you run the benchmarks as above, a comprehensive report for each run will be generated and saved to a JSON file in  `./benchmarks/.benchmarks`. The file name will begin
with `000N` where N is incremented for every run. For example, the report file name for first benchmarks run will look like:
```
0001_f492e1952d5981527d89229e557006a1db992e5f_20230201_230822.json
```

A hook to `pytest-benchmark` was developed to add the following information to the JSON report:

GPUs used by Morpheus. For example:
```
"gpu_0": {
    "id": 0,
    "name": "Quadro RTX 8000",
    "load": "0.0%",
    "free_memory": "47627.0MB",
    "used_memory": "965.0MB",
    "temperature": "55.0 C",
    "uuid": "GPU-6fa37f47-763b-fc49-1b15-75b0d36525bf"
}
```

Morpheus configuration for each workflow:
- `num_threads`
- `pipeline_batch_size`
- `edge_buffer_size`
- `start_time`
- `duration`
- `userid_column_name`
- `timestamp_column_name`
- `source`
- `use_cpp`

Additional benchmark stats for each workflow:
- `input_lines`
- `min_throughput_lines`
- `max_throughput_lines`
- `mean_throughput_lines`
- `median_throughput_lines`
- `input_bytes`
- `min_throughput_bytes`
- `max_throughput_bytes`
- `mean_throughput_bytes`
- `median_throughput_bytes`
