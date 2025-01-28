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

# Real-World Application: Phishing Detection
> **Note**: The code for this guide can be found in the `examples/developer_guide/2_1_real_world_phishing` directory of the Morpheus repository.

## Data Preprocessing

The previous example demonstrated how to create a simple stage and use it in the context of a pipeline, we'll move on to a more advanced example that is representative of what we might want to do in a real-world situation. Given a set of records, each of which represents an email, suppose we want to predict which records correspond to fraudulent emails.

As part of this process, we might want to use a classification model trained on various pieces of metadata, such as recipient count, in addition to the raw content of each email. If we suppose this is true for our example, we need to build and connect a pre-processing stage to attach this information to each record before applying our classifier.

For this task, we'll need to define a new stage, which we will call our `RecipientFeaturesStage`, that will:
1. Receive an input corresponding to an email.
1. Count the number of recipients in the email's metadata.
1. Emit a Morpheus `MessageMeta` object that will contain the record content along with the augmented metadata.

For this stage, the code will be similar to the previous example with a few notable changes. We will be working with the {py:class}`~morpheus.messages.MessageMeta` class. This is a Morpheus message containing a [cuDF](https://docs.rapids.ai/api/cudf/stable/) [DataFrame](https://docs.rapids.ai/api/cudf/stable/user_guide/api_docs/dataframe/). Since we will expect our new stage to operate on `MessageMeta` types, our new `accepted_types` method is defined as:

```python
def accepted_types(self) -> tuple:
    return (MessageMeta, )
```

Next, we will update our `on_data` method to perform the actual work. We grab a reference to the incoming message's `df` attribute. It is important to note that `message` is a reference, and any changes made to it or its members (such as `df`) will be performed in place on the existing message instance.

```python
def on_data(self, message: MessageMeta) -> MessageMeta:
    # Open the DataFrame from the incoming message for in-place modification
    with message.mutable_dataframe() as df:
        df['to_count'] = df['To'].str.count('@')
        df['bcc_count'] = df['BCC'].str.count('@')
        df['cc_count'] = df['CC'].str.count('@')
        df['total_recipients'] = df['to_count'] + df['bcc_count'] + df['cc_count']

        # Attach features to string data
        df['data'] = (df['to_count'].astype(str) + '[SEP]' + df['bcc_count'].astype(str) + '[SEP]' +
                        df['cc_count'].astype(str) + '[SEP]' + df['total_recipients'].astype(str) + '[SEP]' +
                        df['Message'])

    # Return the message for the next stage
    return message
```

If instead mutating the DataFrame in place is undesirable, we could make a copy of the DataFrame with the `MessageMeta.copy_dataframe` method and return a new `MessageMeta`. Note, however, that this would come at the cost of performance and increased memory usage. We could do this by changing the `on_data` method to:
```python
def on_data(self, message: MessageMeta) -> MessageMeta:
    # Get a copy of the DataFrame from the incoming message
    df = message.copy_dataframe()

    df['to_count'] = df['To'].str.count('@')
    df['bcc_count'] = df['BCC'].str.count('@')
    df['cc_count'] = df['CC'].str.count('@')
    df['total_recipients'] = df['to_count'] + df['bcc_count'] + df['cc_count']

    # Attach features to string data
    df['data'] = (df['to_count'].astype(str) + '[SEP]' + df['bcc_count'].astype(str) + '[SEP]' +
                    df['cc_count'].astype(str) + '[SEP]' + df['total_recipients'].astype(str) + '[SEP]' +
                    df['Message'])

    # Return a new message with our updated DataFrame for the next stage
    return MessageMeta(df)
```

In the above example we added five new fields to the DataFrame. Since these fields and their types are known to us ahead of time, as an optimization we can ask Morpheus to pre-allocate these new fields when the DataFrame is first constructed. To do this we populate the `_needed_columns` attribute in our constructor:
```python
from morpheus.common import TypeId

# ...

def __init__(self, config: Config):
    super().__init__(config)

    # This stage adds new columns to the DataFrame, as an optimization we define the columns that are needed,
    # ensuring that these columns are pre-allocated with null values. This action is performed by Morpheus for any
    # stage defining this attribute.
    self._needed_columns.update({
        'to_count': TypeId.INT32,
        'bcc_count': TypeId.INT32,
        'cc_count': TypeId.INT32,
        'total_recipients': TypeId.INT32,
        'data': TypeId.STRING
    })
```

Refer to the [Stage Constructors](#stage-constructors) section for more details.

Since the purpose of this stage is specifically tied to pre-processing text data for an NLP pipeline, when we register the stage, we will explicitly limit the stage to NLP pipelines. In addition to this since the pipeline our stage is operating in is a GPU pipeline, we will not be utilizing the `GpuAndCpuMixin` mixin from the previous example.:
```python
@register_stage("recipient-features", modes=[PipelineModes.NLP])
class RecipientFeaturesStage(PassThruTypeMixin, SinglePortStage):
```

Our `_build_single` method remains unchanged from the previous example; even though we are modifying the incoming messages, our input and output types remain the same and we continue to make use of the {py:class}`~morpheus.pipeline.pass_thru_type_mixin.PassThruTypeMixin`.

### The Completed Preprocessing Stage

```python
import mrc
from mrc.core import operators as ops

from morpheus.cli.register_stage import register_stage
from morpheus.common import TypeId
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.messages.message_meta import MessageMeta
from morpheus.pipeline.pass_thru_type_mixin import PassThruTypeMixin
from morpheus.pipeline.single_port_stage import SinglePortStage


@register_stage("recipient-features", modes=[PipelineModes.NLP])
class RecipientFeaturesStage(PassThruTypeMixin, SinglePortStage):
    """
    Pre-processing stage which counts the number of recipients in an email's metadata.

    Parameters
    ----------
    config : morpheus.config.Config
        Pipeline configuration instance.
    """

    def __init__(self, config: Config):
        super().__init__(config)

        # This stage adds new columns to the DataFrame, as an optimization we define the columns that are needed,
        # ensuring that these columns are pre-allocated with null values. This action is performed by Morpheus for any
        # stage defining this attribute.
        self._needed_columns.update({
            'to_count': TypeId.INT32,
            'bcc_count': TypeId.INT32,
            'cc_count': TypeId.INT32,
            'total_recipients': TypeId.INT32,
            'data': TypeId.STRING
        })

    @property
    def name(self) -> str:
        return "recipient-features"

    def accepted_types(self) -> tuple:
        return (MessageMeta, )

    def supports_cpp_node(self) -> bool:
        return False

    def on_data(self, message: MessageMeta) -> MessageMeta:
        # Open the DataFrame from the incoming message for in-place modification
        with message.mutable_dataframe() as df:
            df['to_count'] = df['To'].str.count('@')
            df['bcc_count'] = df['BCC'].str.count('@')
            df['cc_count'] = df['CC'].str.count('@')
            df['total_recipients'] = df['to_count'] + df['bcc_count'] + df['cc_count']

            # Attach features to string data
            df['data'] = (df['to_count'].astype(str) + '[SEP]' + df['bcc_count'].astype(str) + '[SEP]' +
                          df['cc_count'].astype(str) + '[SEP]' + df['total_recipients'].astype(str) + '[SEP]' +
                          df['Message'])

        # Return the message for the next stage
        return message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
```

### Stand-alone Function

For this example we started with the class based approach. However we could have just as easily written this as a stand-alone function. The following example is equivalent to the class based example above:

```python
from morpheus.common import TypeId
from morpheus.messages import MessageMeta
from morpheus.pipeline.stage_decorator import stage


@stage(
    needed_columns={
        'to_count': TypeId.INT32,
        'bcc_count': TypeId.INT32,
        'cc_count': TypeId.INT32,
        'total_recipients': TypeId.INT32,
        'data': TypeId.STRING
    })
def recipient_features_stage(message: MessageMeta, *, sep_token: str = '[SEP]') -> MessageMeta:
    # Open the DataFrame from the incoming message for in-place modification
    with message.mutable_dataframe() as df:
        df['to_count'] = df['To'].str.count('@')
        df['bcc_count'] = df['BCC'].str.count('@')
        df['cc_count'] = df['CC'].str.count('@')
        df['total_recipients'] = df['to_count'] + df['bcc_count'] + df['cc_count']

        # Attach features to string data
        df['data'] = (df['to_count'].astype(str) + sep_token + df['bcc_count'].astype(str) + sep_token +
                      df['cc_count'].astype(str) + sep_token + df['total_recipients'].astype(str) + sep_token +
                      df['Message'])

    # Return the message for the next stage
    return message
```

In the above the `needed_columns` were provided to as an argument to the `stage` decorator, and the optional `sep_token` argument is exposed as a keyword argument.

> **Note**: One draw-back to the `stage` decorator approach is that we lose the ability to determine the `needed_columns` at runtime based upon constructor arguments.

## Predicting Fraudulent Emails with Accelerated Machine Learning

Now we'll use the `RecipientFeaturesStage` that we just made in a real-world pipeline to detect fraudulent emails. The pipeline we will be building makes use of the `TritonInferenceStage` which is a pre-defined Morpheus stage designed to support the execution of Natural Language Processing (NLP) models via NVIDIA's [Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server). NVIDIA Triton Inference Server allows for GPU accelerated ML/DL and seamless co-location and execution of a wide variety of model frameworks. For our application, we will be using the `phishing-bert-onnx` model, which is included with Morpheus models Docker container as well as in the `models/triton-model-repo/phishing-bert-onnx` directory.

It's important to note here that Triton is a service that is external to the Morpheus pipeline and often will not reside on the same machine as the rest of the pipeline. The `TritonInferenceStage` will use HTTP and [gRPC](https://grpc.io/) network protocols to allow us to interact with the machine learning models that are hosted by the Triton server.

### Launching Triton

Triton will need to be running while we execute our pipeline. For simplicity, we will be using the Morpheus models container which includes both Triton and the Morpheus models.

> **Note**: This step assumes you have both [Docker](https://docs.docker.com/engine/install/) and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installation) installed.

We will launch a Triton Docker container with:

```shell
docker run --rm -ti --gpus=all -p8000:8000 -p8001:8001 -p8002:8002 \
  nvcr.io/nvidia/morpheus/morpheus-tritonserver-models:25.02 \
  tritonserver --model-repository=/models/triton-model-repo \
    --exit-on-error=false \
    --log-info=true \
    --strict-readiness=false \
    --disable-auto-complete-config \
    --model-control-mode=explicit \
    --load-model=phishing-bert-onnx
```

Once we have Triton running, we can verify that it is healthy using [curl](https://curl.se/). The `/v2/health/live` endpoint should return a 200 status code:

```shell
curl -v "localhost:8000/v2/health/live"
```

We can also query Triton for the available models:

```shell
curl -X POST "localhost:8000/v2/repository/index"
```

Let's ask Triton for some information about the `phishing-bert-onnx` model which we are going to be using, parsing the large JSON output with [`jq`](https://stedolan.github.io/jq/):

```shell
curl "localhost:8000/v2/models/phishing-bert-onnx/config" | jq
```

Output:
```json
{
  "name": "phishing-bert-onnx",
  "platform": "onnxruntime_onnx",
  "backend": "onnxruntime",
  "version_policy": {
    "latest": {
      "num_versions": 1
    }
  },
  "max_batch_size": 32,
  "input": [
    {
      "name": "input_ids",
      "data_type": "TYPE_INT64",
      "format": "FORMAT_NONE",
      "dims": [
        128
      ],
      "is_shape_tensor": false,
      "allow_ragged_batch": false,
      "optional": false
    },
    {
      "name": "attention_mask",
      "data_type": "TYPE_INT64",
      "format": "FORMAT_NONE",
      "dims": [
        128
      ],
      "is_shape_tensor": false,
      "allow_ragged_batch": false,
      "optional": false
    }
  ],
  "output": [
    {
      "name": "output",
      "data_type": "TYPE_FP32",
      "dims": [
        2
      ],
      "label_filename": "",
      "is_shape_tensor": false
    }
  ],
  "batch_input": [],
  "batch_output": [],
  "optimization": {
    "priority": "PRIORITY_DEFAULT",
    "execution_accelerators": {
      "gpu_execution_accelerator": [
        {
          "name": "tensorrt",
          "parameters": {
            "max_workspace_size_bytes": "1073741824",
            "precision_mode": "FP16"
          }
        }
      ],
      "cpu_execution_accelerator": []
    },
    "input_pinned_memory": {
      "enable": true
    },
    "output_pinned_memory": {
      "enable": true
    },
    "gather_kernel_buffer_threshold": 0,
    "eager_batching": false
  },
  "dynamic_batching": {
    "preferred_batch_size": [
      1,
      4,
      8,
      12,
      16,
      20,
      24,
      28,
      32
    ],
    "max_queue_delay_microseconds": 50000,
    "preserve_ordering": false,
    "priority_levels": 0,
    "default_priority_level": 0,
    "priority_queue_policy": {}
  },
  "instance_group": [
    {
      "name": "phishing-bert-onnx",
      "kind": "KIND_GPU",
      "count": 1,
      "gpus": [
        0
      ],
      "secondary_devices": [],
      "profile": [],
      "passive": false,
      "host_policy": ""
    }
  ],
  "default_model_filename": "model.onnx",
  "cc_model_filenames": {},
  "metric_tags": {},
  "parameters": {},
  "model_warmup": []
}
```

From this information, we note that the expected dimensions of the model inputs is `"dims": [128]`.

### Defining our Pipeline
For this pipeline we will have several configuration parameters such as the paths to the input and output files, we will be using the (click)[https://click.palletsprojects.com/] library to expose and parse these parameters as command line arguments. We will also expose the choice of using the class or function based stage implementation via the `--use_stage_function` command-line flag.

> **Note**: For simplicity, we assume that the `MORPHEUS_ROOT` environment variable is set to the root of the Morpheus project repository.

To start, we will need to instantiate and set a few attributes of the `Config` class. This object is used for configuration options that are global to the pipeline as a whole. We will provide this object to each stage along with stage-specific configuration parameters.

```python
config = Config()
config.mode = PipelineModes.NLP

config.num_threads = len(os.sched_getaffinity(0))
config.feature_length = model_fea_length

with open(labels_file, encoding='UTF-8') as fh:
    config.class_labels = [x.strip() for x in fh]
```

First we set our pipeline mode to `NLP`. Next, we set the `num_threads` property to match the number of cores in our system.

The `feature_length` property needs to match the dimensions of the model inputs, which we got from Triton in the previous section using the model's `/config` endpoint.

Ground truth classification labels are read from the `morpheus/data/labels_phishing.txt` file included in Morpheus.

Now that our configuration object is populated, we move on to the pipeline itself. We will be using the same input file from the previous example.

Next, we will add our custom recipient features stage to the pipeline. We imported both implementations of the stage, allowing us to add the appropriate one based on the `use_stage_function` value provided by the command-line.

```python
# Add our custom stage
if use_stage_function:
    pipeline.add_stage(recipient_features_stage(config))
else:
    pipeline.add_stage(RecipientFeaturesStage(config))
```

To tokenize the input data we will use Morpheus' `PreprocessNLPStage`. This stage uses the [cuDF subword tokenizer](https://docs.rapids.ai/api/cudf/legacy/user_guide/api_docs/subword_tokenize/#subwordtokenizer) to transform strings into a tensor of numbers to be fed into the neural network model. Rather than split the string by characters or whitespaces, we split them into meaningful subwords based upon the occurrence of the subwords in a large training corpus. You can find more details here: [https://arxiv.org/abs/1810.04805v2](https://arxiv.org/abs/1810.04805v2). All we need to know for now is that the text will be converted to subword token ids based on the vocabulary file that we provide (`vocab_hash_file=vocab file`).

Let's go ahead and instantiate our `PreprocessNLPStage` and add it to the pipeline:

```python
pipeline.add_stage(
    PreprocessNLPStage(
        config,
        vocab_hash_file=vocab_file,
        truncation=True,
        do_lower_case=True,
        add_special_tokens=False))
```

In addition to providing the `Config` object that we defined above, we also configure this stage to:
* Use the `morpheus/data/bert-base-uncased-hash.txt` vocabulary file for its subword token ids (`vocab_hash_file=vocab_file`).
* Truncate the length of the text to a max number of tokens (`truncation=True`).
* Change the casing to all lowercase (`do_lower_case=True`).
* Refrain from adding the default BERT special tokens like `[SEP]` for separation between two sentences and `[CLS]` at the start of the text (`add_special_tokens=False`).

Note that the tokenizer parameters and vocabulary hash file should exactly match what was used for tokenization during the training of the NLP model.

At this point, we have a pipeline that reads in a set of records and pre-processes them with the metadata required for our classifier to make predictions. Our next step is to define a stage that applies a machine learning model to our `MessageMeta` object. To accomplish this, we will be using Morpheus' `TritonInferenceStage`. This stage will handle communication with the `phishing-bert-onnx` model, which we provided to the Triton Docker container via the `models` directory mount.

Next we will add a monitor stage to measure the inference rate:

```python
# Add a inference stage
pipeline.add_stage(
    TritonInferenceStage(
        config,
        model_name=model_name,
        server_url=server_url,
        force_convert_inputs=True,
    ))

pipeline.add_stage(MonitorStage(config, description="Inference Rate", smoothing=0.001, unit="inf"))
```

Here we add a post-processing stage that adds the probability score for `is_phishing`:

```python
pipeline.add_stage(AddScoresStage(config, labels=["is_phishing"]))
```

Lastly, we will save our results to disk. For this purpose, we are using two stages that are often used in conjunction with each other: `SerializeStage` and `WriteToFileStage`.

The `SerializeStage` is used to include and exclude columns as desired in the output. Importantly, it also handles conversion from `ControlMessage` output type to the `MessageMeta` class that is expected as input by the `WriteToFileStage`.

The `WriteToFileStage` will append message data to the output file as messages are received. Note however that for performance reasons the `WriteToFileStage` does not flush its contents out to disk every time a message is received. Instead, it relies on the underlying [buffered output stream](https://gcc.gnu.org/onlinedocs/libstdc++/manual/streambufs.html) to flush as needed, and then will close the file handle on shutdown.

```python
# Write the file to the output
pipeline.add_stage(SerializeStage(config))
pipeline.add_stage(WriteToFileStage(config, filename=output_file, overwrite=True))
```

Note that we didn't specify the output format. In our example, the default output file name contains the extension `.jsonlines`. Morpheus will infer the output format based on the extension. At time of writing the extensions that Morpheus will infer are: `.csv`, `.json` & `.jsonlines`.

To explicitly set the output format we could specify the `file_type` argument to the `WriteToFileStage` which is an enumeration defined in `morpheus.common.FileTypes`. Supported values are:
* `FileTypes.Auto`
* `FileTypes.CSV`
* `FileTypes.JSON`

### The Completed Pipeline

```python
import logging
import os

import click

import morpheus
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.pipeline import LinearPipeline
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.inference.triton_inference_stage import TritonInferenceStage
from morpheus.stages.input.file_source_stage import FileSourceStage
from morpheus.stages.output.write_to_file_stage import WriteToFileStage
from morpheus.stages.postprocess.add_scores_stage import AddScoresStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.stages.preprocess.preprocess_nlp_stage import PreprocessNLPStage
from morpheus.utils.logger import configure_logging
from recipient_features_stage import RecipientFeaturesStage
from recipient_features_stage_deco import recipient_features_stage

MORPHEUS_ROOT = os.environ['MORPHEUS_ROOT']


@click.command()
@click.option("--use_stage_function",
              is_flag=True,
              default=False,
              help="Use the function based version of the recipient features stage instead of the class")
@click.option(
    "--labels_file",
    type=click.Path(exists=True, readable=True),
    default=os.path.join(morpheus.DATA_DIR, 'labels_phishing.txt'),
    help="Specifies a file to read labels from in order to convert class IDs into labels.",
)
@click.option(
    "--vocab_file",
    type=click.Path(exists=True, readable=True),
    default=os.path.join(morpheus.DATA_DIR, 'bert-base-uncased-hash.txt'),
    help="Path to hash file containing vocabulary of words with token-ids.",
)
@click.option(
    "--input_file",
    type=click.Path(exists=True, readable=True),
    default=os.path.join(MORPHEUS_ROOT, 'examples/data/email_with_addresses.jsonlines'),
    help="Input filepath.",
)
@click.option(
    "--model_fea_length",
    default=128,
    type=click.IntRange(min=1),
    help="Features length to use for the model.",
)
@click.option(
    "--model_name",
    default="phishing-bert-onnx",
    help="The name of the model that is deployed on Tritonserver.",
)
@click.option("--server_url", default='localhost:8000', help="Tritonserver url.")
@click.option(
    "--output_file",
    default=".tmp/output/phishing_detections.jsonlines",
    help="The path to the file where the inference output will be saved.",
)
def run_pipeline(use_stage_function: bool,
                 labels_file: str,
                 vocab_file: str,
                 input_file: str,
                 model_fea_length: int,
                 model_name: str,
                 server_url: str,
                 output_file: str):
    """Run the phishing detection pipeline."""
    # Enable the default logger
    configure_logging(log_level=logging.INFO)

    # It's necessary to configure the pipeline for NLP mode
    config = Config()
    config.mode = PipelineModes.NLP

    # Set the thread count to match our cpu count
    config.num_threads = len(os.sched_getaffinity(0))
    config.feature_length = model_fea_length

    with open(labels_file, encoding='UTF-8') as fh:
        config.class_labels = [x.strip() for x in fh]

    # Create a linear pipeline object
    pipeline = LinearPipeline(config)

    # Set source stage
    pipeline.set_source(FileSourceStage(config, filename=input_file, iterative=False))

    # Add our custom stage
    if use_stage_function:
        pipeline.add_stage(recipient_features_stage(config))
    else:
        pipeline.add_stage(RecipientFeaturesStage(config))

    # Add a deserialize stage
    pipeline.add_stage(DeserializeStage(config))

    # Tokenize the input
    pipeline.add_stage(
        PreprocessNLPStage(config,
                           vocab_hash_file=vocab_file,
                           truncation=True,
                           do_lower_case=True,
                           add_special_tokens=False))

    # Add a inference stage
    pipeline.add_stage(
        TritonInferenceStage(
            config,
            model_name=model_name,
            server_url=server_url,
            force_convert_inputs=True,
        ))

    # Monitor the inference rate
    pipeline.add_stage(MonitorStage(config, description="Inference Rate", smoothing=0.001, unit="inf"))

    # Add detection scores
    pipeline.add_stage(AddScoresStage(config, labels=["is_phishing"]))

    # Write the to the output file
    pipeline.add_stage(SerializeStage(config))
    pipeline.add_stage(WriteToFileStage(config, filename=output_file, overwrite=True))

    # Run the pipeline
    pipeline.run()


if __name__ == "__main__":
    run_pipeline()
```

### Alternate Morpheus CLI example
The above pipeline could also be constructed using the Morpheus CLI.

From the root of the Morpheus repo run:
```bash
morpheus --log_level=debug --plugin examples/developer_guide/2_1_real_world_phishing/recipient_features_stage.py \
  run pipeline-nlp --labels_file=data/labels_phishing.txt --model_seq_length=128 \
  from-file --filename=examples/data/email_with_addresses.jsonlines \
  recipient-features \
  deserialize \
  preprocess --vocab_hash_file=data/bert-base-uncased-hash.txt --truncation=true --do_lower_case=true --add_special_tokens=false \
  inf-triton --model_name=phishing-bert-onnx --server_url=localhost:8000 --force_convert_inputs=true \
  monitor --description="Inference Rate" --smoothing=0.001 --unit=inf \
  add-scores --label=is_phishing \
  serialize \
  to-file --filename=.tmp/output/phishing_detections_cli.jsonlines --overwrite
```

## Stage Constructors

In our `RecipientFeaturesStage` example we added a constructor to our stage, however we didn't go into much detail on the implementation. Every stage constructor must receive an instance of a {py:class}`~morpheus.config.Config` object as its first argument and is then free to define additional stage-specific arguments after that. The Morpheus configuration object will contain configuration parameters needed by multiple stages in the pipeline, and the constructor in each Morpheus stage is free to inspect these. In contrast, parameters specific to a single stage are typically defined as constructor arguments. It is a best practice to perform any necessary validation checks in the constructor, and raising an exception in the case of mis-configuration. This allows us to fail early rather than after the pipeline has started.

In our `RecipientFeaturesStage` example, we hard-coded the Bert separator token. Let's instead refactor the code to receive that as a constructor argument. This new constructor argument is documented following the [`numpydoc`](https://numpydoc.readthedocs.io/en/latest/format.html#parameters) formatting style allowing it to be documented properly for both API and CLI users. Let's also take the opportunity to verify that the pipeline mode is set to `morpheus.config.PipelineModes.NLP`.

> **Note**: Setting the pipeline mode in the `register_stage` decorator restricts usage of our stage to NLP pipelines when using the Morpheus command line tool, however there is no such enforcement with the Python API.

Our refactored class definition is now:

```python
@register_stage("recipient-features", modes=[PipelineModes.NLP])
class RecipientFeaturesStage(PassThruTypeMixin, SinglePortStage):
    """
    Pre-processing stage which counts the number of recipients in an email's metadata.

    Parameters
    ----------
    config : morpheus.config.Config
        Pipeline configuration instance.
    sep_token : str
        Bert separator token.
    """

    def __init__(self, config: Config, sep_token: str = '[SEP]'):
        super().__init__(config)
        if config.mode != PipelineModes.NLP:
            raise RuntimeError(
                "RecipientFeaturesStage must be used in a pipeline configured for NLP"
            )

        if len(sep_token) > 0:
            self._sep_token = sep_token
        else:
            raise ValueError("sep_token cannot be an empty string")

        # This stage adds new columns to the DataFrame, as an optimization we define the columns that are needed,
        # ensuring that these columns are pre-allocated with null values. This action is performed by Morpheus for any
        # stage defining this attribute.
        self._needed_columns.update({
            'to_count': TypeId.INT32,
            'bcc_count': TypeId.INT32,
            'cc_count': TypeId.INT32,
            'total_recipients': TypeId.INT32,
            'data': TypeId.STRING
        })

    @property
    def name(self) -> str:
        return "recipient-features"

    def accepted_types(self) -> tuple:
        return (MessageMeta, )

    def supports_cpp_node(self) -> bool:
        return False

    def on_data(self, message: MessageMeta) -> MessageMeta:
        # Open the DataFrame from the incoming message for in-place modification
        with message.mutable_dataframe() as df:
            df['to_count'] = df['To'].str.count('@')
            df['bcc_count'] = df['BCC'].str.count('@')
            df['cc_count'] = df['CC'].str.count('@')
            df['total_recipients'] = df['to_count'] + df['bcc_count'] + df[
                'cc_count']

            # Attach features to string data
            df['data'] = (df['to_count'].astype(str) + self._sep_token +
                          df['bcc_count'].astype(str) + self._sep_token +
                          df['cc_count'].astype(str) + self._sep_token +
                          df['total_recipients'].astype(str) +
                          self._sep_token + df['Message'])

        # Return the message for the next stage
        return message

    def _build_single(self, builder: mrc.Builder,
                      input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
```

If we were to make the above changes, we can view the resulting help string with:
```bash
morpheus --plugin examples/developer_guide/2_1_real_world_phishing/recipient_features_stage.py run pipeline-nlp recipient-features --help
```
```console
Configuring Pipeline via CLI
Usage: morpheus run pipeline-nlp recipient-features [OPTIONS]

  Pre-processing stage which counts the number of recipients in an email's metadata.

Options:
  --sep_token TEXT  Bert separator token.  [default: [SEP]]
  --help            Show this message and exit.
```

## Defining a New Source Stage

> **Note**: The code for this guide can be found in the `examples/developer_guide/2_2_rabbitmq` directory of the Morpheus repository.

### Class Based Approach

Creating a new source stage is similar to defining any other stage with a few differences. First, we will be subclassing {py:class}`~morpheus.pipeline.single_output_source.SingleOutputSource` and including the `PreallocatorMixin`. Second, the required methods are the `name` property, `_build_source`, `compute_schema` and `supports_cpp_node` methods.

In this example, we will create a source that reads messages from a [RabbitMQ](https://www.rabbitmq.com/) queue using the [pika](https://pika.readthedocs.io/en/stable/#) client for Python. For simplicity, we will assume that authentication is not required for our RabbitMQ exchange and that the body of the RabbitMQ messages will be JSON formatted. Both authentication and support for other formats could be easily added later.

The `PreallocatorMixin` when added to a stage class, typically a source stage, indicates that the stage emits newly constructed DataFrames either directly or contained in a `MessageMeta` instance into the pipeline. Adding this mixin allows any columns needed by other stages to be inserted into the DataFrame.

Similar to the pass through stage, this new source stage should be able to operate in both GPU and CPU execution modes, as such we will be using the `GpuAndCpuMixin` mixin. One thing to note is that the DataFrame payload of a `MessageMeta` object is always a `cudf.DataFrame` when running in GPU mode and a `pandas.DataFrame` when running in CPU mode. When supporting both GPU and CPU execution modes, care must be taken to avoid directly importing `cudf` (or any other package requiring a GPU) when running in CPU mode on a system without a GPU and would therefore result in an error. Stages are able to examine the execution mode with the `morpheus.config.Config.execution_mode` attribute. The {py:func}`~morpheus.utils.type_utils.get_df_pkg` helper method is used to import the appropriate DataFrame package based on the execution mode in the constructor:
```python
    # This will return either cudf.DataFrame or pandas.DataFrame depending on the execution mode
    self._df_pkg = get_df_pkg(config.execution_mode)
```

The `compute_schema` method allows us to define our output type of `MessageMeta`, we do so by calling the `set_type` method of the `output_schema` attribute of the `StageSchema` object passed into the method. Of note here is that it is perfectly valid for a stage to determine its output type based upon configuration arguments passed into the constructor. However the stage must document a single output type per output port. If a stage emitted multiple output types, then the types must share a common base class which would serve as the stage's output type.
```python
def compute_schema(self, schema: StageSchema):
    schema.output_schema.set_type(MessageMeta)
```

The `_build_source` method is similar to the `_build_single` method; it receives an instance of the MRC segment builder (`mrc.Builder`) and returns a `mrc.SegmentObject`. However, unlike in the previous examples, source stages do not have a parent stage and therefore do not receive an input node. Instead of building our node with `make_node`, we will call `make_source` with the parameter `self.source_generator`, which is a method that we will define next.

```python
def _build_source(self, builder: mrc.Builder) -> mrc.SegmentObject:
    return builder.make_source(self.unique_name, self.source_generator)
```

The `source_generator` method is where most of the RabbitMQ-specific code exists. When we have a message that we wish to emit into the pipeline, we simply `yield` it. We continue this process until the `is_stop_requested()` method returns `True` or `subscription.is_subscribed()` returns `False`.

```python
def source_generator(self, subscription: mrc.Subscription) -> collections.abc.Iterator[MessageMeta]:
    try:
        while not self.is_stop_requested() and subscription.is_subscribed():
            (method_frame, _, body) = self._channel.basic_get(self._queue_name)
            if method_frame is not None:
                try:
                    buffer = StringIO(body.decode("utf-8"))
                    df = self._df_pkg.read_json(buffer, orient='records', lines=True)
                    yield MessageMeta(df=df)
                except Exception as ex:
                    logger.exception("Error occurred converting RabbitMQ message to Dataframe: %s", ex)
                finally:
                    self._channel.basic_ack(method_frame.delivery_tag)
            else:
                # queue is empty, sleep before polling again
                time.sleep(self._poll_interval.total_seconds())

    finally:
        self._connection.close()
```

Note that we read messages as quickly as we can from the queue. When the queue is empty we call `time.sleep`, allowing for a context switch to occur if needed. We acknowledge the message (by calling `basic_ack`) only once we have successfully emitted the message or failed to deserialize the message. This means that if the pipeline shuts down while consuming the queue, we will not lose any messages. However, in that situation we may end up with a duplicate message (that is, if the pipeline is shut down after we have yielded the message but before calling `basic_ack`).

#### The Completed Source Stage

```python
import collections.abc
import logging
import time
from io import StringIO

import mrc
import pandas as pd
import pika

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.messages.message_meta import MessageMeta
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.preallocator_mixin import PreallocatorMixin
from morpheus.pipeline.single_output_source import SingleOutputSource
from morpheus.pipeline.stage_schema import StageSchema
from morpheus.utils.type_utils import get_df_pkg

logger = logging.getLogger(__name__)


@register_stage("from-rabbitmq")
class RabbitMQSourceStage(PreallocatorMixin, GpuAndCpuMixin, SingleOutputSource):
    """
    Source stage used to load messages from a RabbitMQ queue.

    Parameters
    ----------
    config : morpheus.config.Config
        Pipeline configuration instance.
    host : str
        Hostname or IP of the RabbitMQ server.
    exchange : str
        Name of the RabbitMQ exchange to connect to.
    exchange_type : str, optional
        RabbitMQ exchange type; defaults to `fanout`.
    queue_name : str, optional
        Name of the queue to listen to. If left blank, RabbitMQ will generate a random queue name bound to the exchange.
    poll_interval : str, optional
        Amount of time  between polling RabbitMQ for new messages
    """

    def __init__(self,
                 config: Config,
                 host: str,
                 exchange: str,
                 exchange_type: str = 'fanout',
                 queue_name: str = '',
                 poll_interval: str = '100millis'):
        super().__init__(config)
        self._connection = pika.BlockingConnection(pika.ConnectionParameters(host=host))

        self._channel = self._connection.channel()
        self._channel.exchange_declare(exchange=exchange, exchange_type=exchange_type)

        result = self._channel.queue_declare(queue=queue_name, exclusive=True)

        # When queue_name='' we will receive a randomly generated queue name
        self._queue_name = result.method.queue

        self._channel.queue_bind(exchange=exchange, queue=self._queue_name)

        self._poll_interval = pd.Timedelta(poll_interval)

        # This will return either cudf.DataFrame or pandas.DataFrame depending on the execution mode
        self._df_pkg = get_df_pkg(config.execution_mode)

    @property
    def name(self) -> str:
        return "from-rabbitmq"

    def supports_cpp_node(self) -> bool:
        return False

    def compute_schema(self, schema: StageSchema):
        schema.output_schema.set_type(MessageMeta)

    def _build_source(self, builder: mrc.Builder) -> mrc.SegmentObject:
        return builder.make_source(self.unique_name, self.source_generator)

    def source_generator(self, subscription: mrc.Subscription) -> collections.abc.Iterator[MessageMeta]:
        try:
            while not self.is_stop_requested() and subscription.is_subscribed():
                (method_frame, _, body) = self._channel.basic_get(self._queue_name)
                if method_frame is not None:
                    try:
                        buffer = StringIO(body.decode("utf-8"))
                        df = self._df_pkg.read_json(buffer, orient='records', lines=True)
                        yield MessageMeta(df=df)
                    except Exception as ex:
                        logger.exception("Error occurred converting RabbitMQ message to Dataframe: %s", ex)
                    finally:
                        self._channel.basic_ack(method_frame.delivery_tag)
                else:
                    # queue is empty, sleep before polling again
                    time.sleep(self._poll_interval.total_seconds())

        finally:
            self._connection.close()
```

### Function Based Approach
Similar to the `stage` decorator used in previous examples Morpheus provides a {py:func}`~morpheus.pipeline.stage_decorator.source` decorator which wraps a generator function to be used as a source stage. In the class based approach we explicitly added the `PreallocatorMixin`, when using the `source` decorator the return type annotation will be inspected and a stage will be created with the `PreallocatorMixin` if the return type is a `DataFrame` type or a message which contains a `DataFrame` (`MessageMeta` and `ControlMessage`). We will also indicate which execution modes are supported by the stage by setting the `execution_modes` argument to the decorator.

The code for the function will first perform the same setup as was used in the class constructor, then entering a nearly identical loop as that in the `source_generator` method.

```python
import collections.abc
import logging
import time
from io import StringIO

import mrc
import pandas as pd
import pika

from morpheus.config import ExecutionMode
from morpheus.messages.message_meta import MessageMeta
from morpheus.pipeline.stage_decorator import source
from morpheus.utils.type_utils import get_df_pkg

logger = logging.getLogger(__name__)


@source(name="from-rabbitmq", execution_modes=(ExecutionMode.GPU, ExecutionMode.CPU))
def rabbitmq_source(subscription: mrc.Subscription,
                    host: str,
                    exchange: str,
                    exchange_type: str = 'fanout',
                    queue_name: str = '',
                    poll_interval: str = '100millis') -> collections.abc.Iterator[MessageMeta]:
    """
    Source stage used to load messages from a RabbitMQ queue.

    Parameters
    ----------
    subscription : mrc.Subscription
        Subscription object used to determine if the pipeline is still running.
    host : str
        Hostname or IP of the RabbitMQ server.
    exchange : str
        Name of the RabbitMQ exchange to connect to.
    exchange_type : str, optional
        RabbitMQ exchange type; defaults to `fanout`.
    queue_name : str, optional
        Name of the queue to listen to. If left blank, RabbitMQ will generate a random queue name bound to the exchange.
    poll_interval : str, optional
        Amount of time  between polling RabbitMQ for new messages
    """
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=host))

    channel = connection.channel()
    channel.exchange_declare(exchange=exchange, exchange_type=exchange_type)

    result = channel.queue_declare(queue=queue_name, exclusive=True)

    # When queue_name='' we will receive a randomly generated queue name
    queue_name = result.method.queue

    channel.queue_bind(exchange=exchange, queue=queue_name)

    poll_interval = pd.Timedelta(poll_interval)

    df_pkg = get_df_pkg()

    try:
        while subscription.is_subscribed():
            (method_frame, _, body) = channel.basic_get(queue_name)
            if method_frame is not None:
                try:
                    buffer = StringIO(body.decode("utf-8"))
                    df = df_pkg.read_json(buffer, orient='records', lines=True)
                    yield MessageMeta(df=df)
                except Exception as ex:
                    logger.exception("Error occurred converting RabbitMQ message to Dataframe: %s", ex)
                finally:
                    channel.basic_ack(method_frame.delivery_tag)
            else:
                # queue is empty, sleep before polling again
                time.sleep(poll_interval.total_seconds())

    finally:
        connection.close()
```

## Defining a New Sink Stage

In Morpheus, we define a stage to be a sink if it outputs the results of a pipeline to a destination external to the pipeline. Morpheus includes several sink stages under the `morpheus.stages.output` namespace.

Recall that in the previous section we wrote a `RabbitMQSourceStage`. We will now complement that by writing a sink stage that can output Morpheus data into [RabbitMQ](https://www.rabbitmq.com/). For this example, we are again using the [pika](https://pika.readthedocs.io/en/stable/#) client for Python.

The code for our sink will be similar to other stages with a few changes. First, we will subclass `SinglePortStage`:

```python
@register_stage("to-rabbitmq")
class WriteToRabbitMQStage(PassThruTypeMixin, GpuAndCpuMixin, SinglePortStage):
```

Our sink will function as a pass-through allowing the possibility of other sinks to be added to the pipeline. We could, hypothetically, have a pipeline where we emit the results to both RabbitMQ and a file. For this reason we will also be using the `PassThruTypeMixin`.

![Morpheus node dependency diagram](img/sink_deps.png)

In our `_build_single` method we will be making use of the `make_sink` method rather than `make_node` or `make_source`.
```python
def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
    node = builder.make_sink(self.unique_name, self.on_data, self.on_error, self.on_complete)
    builder.make_edge(input_node, node)
    return node
```

Similar to our previous examples, most of the actual business logic of the stage is contained in the `on_data` method. In this case, we grab a reference to the DataFrane attached to the incoming message. We then serialize to an [`io.StringIO`](https://docs.python.org/3.10/library/io.html?highlight=stringio#io.StringIO) buffer, which is then sent to RabbitMQ.

> **Note**: This stage supports both GPU and CPU execution modes. When running in GPU mode, the payload of a `MessageMeta` object is always a [cuDF](https://docs.rapids.ai/api/cudf/stable/) [DataFrame](https://docs.rapids.ai/api/cudf/stable/user_guide/api_docs/dataframe/). When running in CPU mode, the payload is always a [pandas](https://pandas.pydata.org/) [DataFrane](https://pandas.pydata.org/docs/reference/frame.html). In many cases the two will be API compatible without requiring any changes to the code. In some cases however, the API may differ slightly and there is a need to know the payload type, care must be taken not to directly import `cudf` or any other package requiring a GPU when running in CPU mode on a system without a GPU. Morpheus provides some helper methods to assist with this in the {py:mod}`~morpheus.utils.type_utils` module, such as {py:func}`~morpheus.utils.type_utils.is_cudf_type` and {py:func}`~morpheus.utils.type_utils.get_df_pkg_from_obj`.

```python
    def on_data(self, message: MessageMeta) -> MessageMeta:
        df = message.df

        buffer = StringIO()
        df.to_json(buffer, orient='records', lines=True)
        body = buffer.getvalue().strip()

        self._channel.basic_publish(exchange=self._exchange, routing_key=self._routing_key, body=body)

        return message
```

The two new methods introduced in this example are the `on_error` and `on_complete` methods. For both methods, we want to make sure  the [connection](https://pika.readthedocs.io/en/stable/modules/connection.html) object is properly closed.

> **Note**: We didn't close the channel object since closing the connection will also close any associated channel objects.

```python
def on_error(self, ex: Exception):
    logger.exception("Error occurred : %s", ex)
    self._connection.close()

def on_complete(self):
    self._connection.close()
```

### The Completed Sink Stage

```python
import logging
from io import StringIO

import mrc
import pika

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.messages.message_meta import MessageMeta
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.pass_thru_type_mixin import PassThruTypeMixin
from morpheus.pipeline.single_port_stage import SinglePortStage

logger = logging.getLogger(__name__)


@register_stage("to-rabbitmq")
class WriteToRabbitMQStage(PassThruTypeMixin, GpuAndCpuMixin, SinglePortStage):
    """
    Source stage used to load messages from a RabbitMQ queue.

    Parameters
    ----------
    config : morpheus.config.Config
        Pipeline configuration instance.
    host : str
        Hostname or IP of the RabbitMQ server.
    exchange : str
        Name of the RabbitMQ exchange to connect to.
    exchange_type : str
        RabbitMQ exchange type; defaults to `fanout`.
    routing_key : str
        RabbitMQ routing key if needed.
    """

    def __init__(self, config: Config, host: str, exchange: str, exchange_type: str = 'fanout', routing_key: str = ''):
        super().__init__(config)
        self._connection = pika.BlockingConnection(pika.ConnectionParameters(host=host))

        self._exchange = exchange
        self._routing_key = routing_key

        self._channel = self._connection.channel()
        self._channel.exchange_declare(exchange=self._exchange, exchange_type=exchange_type)

    @property
    def name(self) -> str:
        return "to-rabbitmq"

    def accepted_types(self) -> tuple:
        return (MessageMeta, )

    def supports_cpp_node(self) -> bool:
        return False

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_sink(self.unique_name, self.on_data, self.on_error, self.on_complete)
        builder.make_edge(input_node, node)
        return node

    def on_data(self, message: MessageMeta) -> MessageMeta:
        df = message.df

        buffer = StringIO()
        df.to_json(buffer, orient='records', lines=True)
        body = buffer.getvalue().strip()

        self._channel.basic_publish(exchange=self._exchange, routing_key=self._routing_key, body=body)

        return message

    def on_error(self, ex: Exception):
        logger.exception("Error occurred : %s", ex)
        self._connection.close()

    def on_complete(self):
        self._connection.close()
```

> **Note**: For information about testing the `RabbitMQSourceStage`, `rabbitmq_source`, and `WriteToRabbitMQStage` stages refer to `examples/developer_guide/2_2_rabbitmq/README.md` in the Morpheus repo.
