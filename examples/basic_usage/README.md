<!--
SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Basic Usage Examples

## Simple Identity

This example will copy the values from Kafka into `out.jsonlines`.

![img/simple_identity.png](img/simple_identity.png)

```bash
morpheus run pipeline-nlp --viz_file=basic_usage_img/simple_identity.png  \
   from-kafka --bootstrap_servers localhost:9092 --input_topic test_pcap \
   deserialize \
   serialize \
   to-file --overwrite --filename out.jsonlines
```

## Remove Fields from JSON Objects

This example will only copy the fields 'timestamp', 'src_ip' and 'dest_ip' from `examples/data/pcap_dump.jsonlines` to
`out.jsonlines`.

![img/remove_fields_from_json_objects.png](img/remove_fields_from_json_objects.png)

```bash
morpheus run pipeline-nlp --viz_file=basic_usage_img/remove_fields_from_json_objects.png \
   from-file --filename examples/data/pcap_dump.jsonlines \
   deserialize \
   serialize --include 'timestamp' --include 'src_ip' --include 'dest_ip' \
   to-file --overwrite --filename out.jsonlines
```

## Monitor Throughput

This example will report the throughput on the command line.

![img/monitor_throughput.png](img/monitor_throughput.png)

```bash
morpheus run pipeline-nlp --viz_file=basic_usage_img/monitor_throughput.png  \
   from-file --filename examples/data/pcap_dump.jsonlines \
   deserialize \
   monitor --description "Lines Throughput" --smoothing 0.1 --unit "lines" \
   serialize \
   to-file --overwrite --filename out.jsonlines
```

Output:
```console
Configuring Pipeline via CLI
Starting pipeline via CLI... Ctrl+C to Quit
Lines Throughput[Complete]: 93085 lines [00:04, 19261.06 lines/s]
Pipeline visualization saved to basic_usage_img/monitor_throughput.png
```

## Multi-Monitor Throughput

This example will report the throughput for each stage independently.

![img/multi_monitor_throughput.png](img/multi_monitor_throughput.png)

```bash
morpheus run pipeline-nlp --viz_file=basic_usage_img/multi_monitor_throughput.png  \
   from-file --filename examples/data/pcap_dump.jsonlines \
   monitor --description "From File Throughput" \
   deserialize \
   monitor --description "Deserialize Throughput" \
   serialize \
   monitor --description "Serialize Throughput" \
   to-file --filename out.jsonlines --overwrite
```

Output:
```console
Configuring Pipeline via CLI
Starting pipeline via CLI... Ctrl+C to Quit
From File Throughput[Complete]: 93085 messages [00:00, 93852.05 messages/s]
Deserialize Throughput[Complete]: 93085 messages [00:05, 16898.32 messages/s]
Serialize Throughput[Complete]: 93085 messages [00:08, 11110.10 messages/s]
Pipeline visualization saved to basic_usage_img/multi_monitor_throughput.png
```

## NLP Kitchen Sink

This example shows an NLP Pipeline which uses most stages available in Morpheus.

![img/nlp_kitchen_sink.png](img/nlp_kitchen_sink.png)

```bash
morpheus run --num_threads=8 --pipeline_batch_size=1024 --model_max_batch_size=32 \
   pipeline-nlp --viz_file=basic_usage_img/nlp_kitchen_sink.png  \
   from-file --filename examples/data/pcap_dump.jsonlines \
   deserialize \
   preprocess \
   inf-triton --model_name=sid-minibert-onnx --server_url=localhost:8001 \
   monitor --description "Inference Rate" --smoothing=0.001 --unit "inf" \
   add-class \
   filter --threshold=0.8 \
   serialize --include 'timestamp' --exclude '^_ts_' \
   to-kafka --bootstrap_servers localhost:9092 --output_topic "inference_output" \
   monitor --description "ToKafka Rate" --smoothing=0.001 --unit "msg"
```

Output:
```console
Configuring Pipeline via CLI
Starting pipeline via CLI... Ctrl+C to Quit
Inference Rate[Complete]: 93085 inf [00:07, 12334.49 inf/s]
ToKafka Rate[Complete]: 93085 msg [00:07, 13297.85 msg/s]
Pipeline visualization saved to basic_usage_img/nlp_kitchen_sink.png
```
