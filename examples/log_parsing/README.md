<!--
# Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

# Example cyBERT Morpheus Pipeline for Apache Log Parsing

Example Morpheus pipeline using Triton Inference server and Morpheus.

### Set up Triton Inference Server

##### Pull Triton Inference Server Docker Image
Pull Docker image from NGC (https://ngc.nvidia.com/catalog/containers/nvidia:tritonserver) suitable for your environment.

Example:

```bash
docker pull nvcr.io/nvidia/tritonserver:24.04-py3
```

##### Start Triton Inference Server Container
From the Morpheus repo root directory, run the following to launch Triton and load the `log-parsing-onnx` model:

```bash
docker run --rm -ti --gpus=all -p8000:8000 -p8001:8001 -p8002:8002 -v $PWD/models:/models nvcr.io/nvidia/tritonserver:24.04-py3 tritonserver --model-repository=/models/triton-model-repo --exit-on-error=false --model-control-mode=explicit --load-model log-parsing-onnx
```

##### Verify Model Deployment
Once Triton server finishes starting up, it will display the status of all loaded models. Successful deployment of the model will show the following:

```
+------------------+---------+--------+
| Model            | Version | Status |
+------------------+---------+--------+
| log-parsing-onnx | 1       | READY  |
+------------------+---------+--------+
```

> **Note**: If this is not present in the output, check the Triton log for any error messages related to loading the model.

### Run Log Parsing Pipeline

Run the following from the root of the Morpheus repo to start the log parsing pipeline:

```bash
python examples/log_parsing/run.py \
    --input_file=./models/datasets/validation-data/log-parsing-validation-data-input.csv \
    --model_vocab_hash_file=data/bert-base-cased-hash.txt \
    --model_vocab_file=./models/training-tuning-scripts/sid-models/resources/bert-base-cased-vocab.txt \
    --model_name log-parsing-onnx \
    --model_config_file=./models/log-parsing-models/log-parsing-config-20220418.json
```

Use `--help` to display information about the command line options:

```bash
python run.py --help

Options:
  --num_threads INTEGER RANGE     Number of internal pipeline threads to use
                                  [x>=1]
  --pipeline_batch_size INTEGER RANGE
                                  Internal batch size for the pipeline. Can be
                                  much larger than the model batch size. Also
                                  used for Kafka consumers  [x>=1]
  --model_max_batch_size INTEGER RANGE
                                  Max batch size to use for the model  [x>=1]
  --input_file PATH               Input filepath  [required]
  --output_file TEXT              The path to the file where the inference
                                  output will be saved.
  --model_vocab_hash_file FILE    Model vocab hash file to use for pre-
                                  processing  [required]
  --model_vocab_file FILE         Model vocab file to use for post-processing
                                  [required]
  --model_seq_length INTEGER RANGE
                                  Sequence length to use for the model  [x>=1]
  --model_name TEXT               The name of the model that is deployed on
                                  Triton server  [required]
  --model_config_file TEXT        Model config file  [required]
  --server_url TEXT               Tritonserver url  [required]
  --help                          Show this message and exit.
```

### CLI Example
The above example is illustrative of using the Python API to build a custom Morpheus pipeline. Alternately, the Morpheus command line could have been used to accomplish the same goal. To do this we must ensure the `examples/log_parsing` directory is available in the `PYTHONPATH` and each of the custom stages are registered as plugins.

From the root of the Morpheus repo, run:
```bash
PYTHONPATH="examples/log_parsing" \
morpheus --log_level INFO \
	--plugin "inference" \
	--plugin "postprocessing" \
	run --pipeline_batch_size 1024 --model_max_batch_size 32  \
	pipeline-nlp \
	from-file --filename ./models/datasets/validation-data/log-parsing-validation-data-input.csv  \
	deserialize \
	preprocess --vocab_hash_file data/bert-base-cased-hash.txt --stride 64 --column=raw \
	monitor --description "Preprocessing rate" \
	inf-logparsing --model_name log-parsing-onnx --server_url localhost:8001 --force_convert_inputs=True \
	monitor --description "Inference rate" --unit inf \
	log-postprocess --vocab_path ./models/training-tuning-scripts/sid-models/resources/bert-base-cased-vocab.txt \
		--model_config_path=./models/log-parsing-models/log-parsing-config-20220418.json \
	to-file --filename ./log-parsing-output.jsonlines --overwrite  \
	monitor --description "Postprocessing rate"
```
