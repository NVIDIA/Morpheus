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

# Running E2E Benchmarks

### Set up Triton Inference Server

##### Pull Triton Inference Server Docker Image
Pull Morpheus Models Docker image from NGC.

Example:

```bash
docker pull nvcr.io/nvidia/morpheus/morpheus-tritonserver-models:25.02
```

##### Start Triton Inference Server container
```bash
docker run --gpus=all --rm -p8000:8000 -p8001:8001 -p8002:8002 nvcr.io/nvidia/morpheus/morpheus-tritonserver-models:25.02 tritonserver --model-repository=/models/triton-model-repo --model-control-mode=explicit --load-model sid-minibert-onnx --load-model abp-nvsmi-xgb --load-model phishing-bert-onnx --load-model all-MiniLM-L6-v2
```

##### Verify Model Deployments
Once Triton server finishes starting up, it will display the status of all loaded models. Successful deployment of the model will show the following:

```
+--------------------+---------+--------+
| Model              | Version | Status |
+--------------------+---------+--------+
| abp-nvsmi-xgb      | 1       | READY  |
| all-MiniLM-L6-v2   | 1       | READY  |
| phishing-bert-onnx | 1       | READY  |
| sid-minibert-onnx  | 1       | READY  |
+--------------------+---------+--------+
```

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

Now install Morpheus:
```bash
pip install -e /workspace
```

Fetch input data for benchmarks:
```bash
./scripts/fetch_data.py fetch validation
```

### Run E2E Benchmarks

Benchmarks are run using `pytest-benchmark`. By default, there are five rounds of measurement. For each round, there will be one iteration of each workflow. Measurements are taken for each round. Final results such as `min`, `max` and `mean` times will be based on these measurements.

To provide your own calibration or use other `pytest-benchmark` features with these workflows, please refer to their [documentation](https://pytest-benchmark.readthedocs.io/en/latest/).

Morpheus configurations for each workflow are managed using `e2e_test_configs.json`. For example, this is the Morpheus configuration for  `sid_nlp`:
```
"test_sid_nlp_e2e": {
    "file_path": "../../models/datasets/validation-data/sid-validation-data.csv",
    "repeat": 10,
    "num_threads": 8,
    "pipeline_batch_size": 1024,
    "model_max_batch_size": 64,
    "feature_length": 256,
    "edge_buffer_size": 4
},
...
```

To run all benchmarks run the following:
```bash
cd tests/benchmarks

pytest -s --run_benchmark --run_milvus --benchmark-enable --benchmark-warmup=on --benchmark-warmup-iterations=1 --benchmark-autosave
```

The `-s` option allows outputs of pipeline execution to be displayed so you can ensure there are no errors while running your benchmarks.

The `--benchmark-warmup` and `--benchmark-warmup-iterations` options are used to run the workflows once before starting measurements. This is because the models deployed to Triton are configured to convert from ONNX to TensorRT on first use. Since the conversion can take a considerable amount of time, we don't want to include it in the measurements. The `--run_milvus` flag enables benchmarks which require the Milvus database.

#### Running with an existing Milvus database

By default when `--run_milvus` flag is provided, pytest will start a new Milvus database. If you wish to use an existing Milvus database, you can set the `MORPHEUS_MILVUS_URI` environment variable. For a local Milvus database, running on the default port, you can set the environment variable as follows:
```bash
export MORPHEUS_MILVUS_URI="http://127.0.0.1:19530"
```

#### test_bench_e2e_pipelines.py

The `test_bench_e2e_pipelines.py` script contains several benchmarks within it.
`<test-workflow>` is the name of the test to run benchmarks on. This can be one of the following:
- `test_sid_nlp_e2e`
- `test_abp_fil_e2e`
- `test_phishing_nlp_e2e`

For example, to run E2E benchmarks on the SID NLP workflow:
```bash
pytest -s --run_benchmark --benchmark-enable --benchmark-warmup=on --benchmark-warmup-iterations=1 --benchmark-autosave test_bench_e2e_pipelines.py::test_sid_nlp_e2e
```

To run E2E benchmarks on all workflows:
```bash
pytest -s --run_benchmark --benchmark-enable --benchmark-warmup=on --benchmark-warmup-iterations=1 --benchmark-autosave test_bench_e2e_pipelines.py
```

The console output should look like this:
```
--------------------------------------------------------------------------------- benchmark: 3 tests --------------------------------------------------------------------------------
Name (time in s)              Min               Max              Mean            StdDev            Median               IQR            Outliers     OPS            Rounds  Iterations
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_sid_nlp_e2e           1.8907 (1.0)      1.9817 (1.0)      1.9400 (1.0)      0.0325 (2.12)     1.9438 (1.0)      0.0297 (1.21)          2;0  0.5155 (1.0)           5           1
test_abp_fil_e2e           5.1271 (2.71)     5.3044 (2.68)     5.2083 (2.68)     0.0856 (5.59)     5.1862 (2.67)     0.1653 (6.75)          1;0  0.1920 (0.37)          5           1
test_phishing_nlp_e2e      5.6629 (3.00)     6.0987 (3.08)     5.8835 (3.03)     0.1697 (11.08)    5.8988 (3.03)     0.2584 (10.55)         2;0  0.1700 (0.33)          5           1
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```

### Benchmarks Report

Each time you run the benchmarks as above, a comprehensive report for each run will be generated and saved to a JSON file in  `./tests/benchmarks/.benchmarks`. The file name will begin
with `000N` where N is incremented for every run. For example, the report file name for first benchmarks run will look like:
```
0001_dacccac5198c7eeddc477794bc278028e739c2cd_20220929_182030.json
```

A hook to `pytest-benchmark` was developed to add the following information to the JSON report:

GPUs used by Morpheus. For example:
```
"gpu_0": {
    "id": 0,
    "name": "Quadro RTX 8000",
    "load": "0.0%",
    "free_memory": "42444.0MB",
    "used_memory": "6156.0MB",
    "temperature": "61.0 C",
    "uuid": "GPU-dc32de82-bdaa-2d05-2abe-260a847e1989"
}
```

Morpheus configuration for each workflow:
- `num_threads`
- `pipeline_batch_size`
- `model_max_batch_size`
- `feature_length`
- `edge_buffer_size`

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


### Production DFP E2E Benchmarks

Separate benchmark tests are provided to measure performance of the example [Production DFP](../../examples/digital_fingerprinting/production/README.md) pipelines. More information about running those benchmarks can be found [here](../../examples/digital_fingerprinting/production/morpheus/benchmarks/README.md).

You can use the same Dev container created here to run the Production DFP benchmarks. You would just need to install additional dependencies as follows:

```bash
conda env update --solver=libmamba \
  -n ${CONDA_DEFAULT_ENV} \
  --file ./conda/environments/examples_cuda-125_arch-$(arch).yaml
```
