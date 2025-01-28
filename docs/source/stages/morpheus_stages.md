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

# Stages Documentation

Stages are the building blocks of Morpheus pipelines. Below is a list of the most commonly used stages. For a full list of stages, refer to the stages API {py:mod}`morpheus.stages`. In addition to this there are several custom stages contained in the [Examples](../examples.md) and [Developer Guides](../developer_guide/guides.md).

## Table of Contents
- [DOCA](#doca)
- [General](#general)
- [Inference](#inference)
- [Input](#input)
- [LLM](#llm)
- [Output](#output)
- [Post-process](#post-process)
- [Pre-process](#pre-process)


## DOCA

- DOCA Source Stage {py:class}`~morpheus.stages.doca.doca_source_stage.DocaSourceStage` A source stage used to receive raw packet data in GPU memory from a ConnectX NIC using DOCA GPUNetIO function within a CUDA kernel to actually receive and process Ethernet network packets. Receive packets information is passed to next pipeline stage in the form of RawPacketMessage. This stage is not compiled by default refer to the [DOCA Example](../../../examples/doca/README.md) for details on building this stage.
- DOCA Convert Stage {py:class}`~morpheus.stages.doca.doca_source_stage.DocaConvertStage` Convert the RawPacketMessage format received by the DOCA Source Stage into a more complex message format MetaMessage. Packets' info never leave the GPU memory. This stage is not compiled by default refer to the [DOCA Example](../../../examples/doca/README.md) for details on building this stage.

## General

- Linear Modules Stage {py:class}`~morpheus.stages.general.linear_modules_stage.LinearModulesStage` Loads an existing, registered, module and wraps it as a Morpheus stage. Refer to [Morpheus Modules](../developer_guide/guides.md#morpheus-modules) for details on modules.
- Monitor Stage {py:class}`~morpheus.stages.general.monitor_stage.MonitorStage` Display throughput numbers at a specific point in the pipeline.
- Multi Port Module Stage {py:class}`~morpheus.stages.general.multi_port_modules_stage.MultiPortModulesStage` Loads an existing, registered, multi-port module and wraps it as a multi-port Morpheus stage. Refer to [Morpheus Modules](../developer_guide/guides.md#morpheus-modules) for details on modules.
- Trigger Stage {py:class}`~morpheus.stages.general.trigger_stage.TriggerStage` Buffer data until the previous stage has completed, useful for testing performance of one stage at a time.

## Inference

- PyTorch Inference Stage {py:class}`~morpheus.stages.inference.pytorch_inference_stage.PyTorchInferenceStage` PyTorch inference stage used for most pipeline modes with the exception of Auto Encoder.
- Triton Inference Stage {py:class}`~morpheus.stages.inference.triton_inference_stage.TritonInferenceStage`  Inference stage which utilizes a [Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server).

## Input

- App Shield Source Stage {py:class}`~morpheus.stages.input.appshield_source_stage.AppShieldSourceStage` Load App Shield messages from one or more plugins into a DataFrame.
- Control Message File Source Stage {py:class}`~morpheus.stages.input.control_message_file_source_stage.ControlMessageFileSourceStage` Receives control messages from different sources specified by a list of (fsspec)[https://filesystem-spec.readthedocs.io/en/latest/api.html?highlight=open_files#fsspec.open_files] strings.
- Control Message Kafka Source Stage {py:class}`~morpheus.stages.input.control_message_kafka_source_stage.ControlMessageKafkaSourceStage` Load control messages from a Kafka cluster.
- Databricks Delta Lake Source Stage {py:class}`~morpheus.stages.input.databricks_deltalake_source_stage.DataBricksDeltaLakeSourceStage` Source stage used to load messages from a DeltaLake table.
- File Source Stage {py:class}`~morpheus.stages.input.file_source_stage.FileSourceStage` Load messages from a file.
- HTTP Client Source Stage {py:class}`~morpheus.stages.input.http_client_source_stage.HttpClientSourceStage` Poll a remote HTTP server for incoming data.
- HTTP Server Source Stage {py:class}`~morpheus.stages.input.http_server_source_stage.HttpServerSourceStage` Start an HTTP server and listens for incoming requests on a specified endpoint.
- In Memory Source Stage {py:class}`~morpheus.stages.input.in_memory_source_stage.InMemorySourceStage` Input source that emits a pre-defined list of DataFrames.
- Kafka Source Stage {py:class}`~morpheus.stages.input.kafka_source_stage.KafkaSourceStage` Load messages from a Kafka cluster.
- RSS Source Stage {py:class}`~morpheus.stages.input.rss_source_stage.RSSSourceStage` Load RSS feed items into a pandas DataFrame.

## LLM

- LLM Engine Stage {py:class}`~morpheus_llm.stages.llm.llm_engine_stage.LLMEngineStage` Execute an LLM engine within a Morpheus pipeline.

## Output
- HTTP Client Sink Stage {py:class}`~morpheus.stages.output.http_client_sink_stage.HttpClientSinkStage` Write all messages to an HTTP endpoint.
- HTTP Server Sink Stage {py:class}`~morpheus.stages.output.http_server_sink_stage.HttpServerSinkStage` Start an HTTP server and listens for incoming requests on a specified endpoint.
- In Memory Sink Stage {py:class}`~morpheus.stages.output.in_memory_sink_stage.InMemorySinkStage` Collect incoming messages into a list that can be accessed after the pipeline is complete.
- Databricks Delta Lake Sink Stage {py:class}`~morpheus.stages.output.write_to_databricks_deltalake_stage.DataBricksDeltaLakeSinkStage` Write messages to a DeltaLake table.
- Write To Elastic Search Stage {py:class}`~morpheus.stages.output.write_to_elasticsearch_stage.WriteToElasticsearchStage` Write the messages as documents to Elasticsearch.
- Write To File Stage {py:class}`~morpheus.stages.output.write_to_file_stage.WriteToFileStage` Write all messages to a file.
- Write To Kafka Stage {py:class}`~morpheus.stages.output.write_to_kafka_stage.WriteToKafkaStage` Write all messages to a Kafka cluster.
- Write To Vector DB Stage {py:class}`~morpheus.stages.output.write_to_vector_db.WriteToVectorDBStage` Write all messages to a Vector Database.

## Post-process

- Add Classifications Stage {py:class}`~morpheus.stages.postprocess.add_classifications_stage.AddClassificationsStage` Add detected classifications to each message.
- Add Scores Stage {py:class}`~morpheus.stages.postprocess.add_scores_stage.AddScoresStage` Add probability scores to each message.
- Filter Detections Stage {py:class}`~morpheus.stages.postprocess.filter_detections_stage.FilterDetectionsStage` Filter message by a classification threshold.
- Generate Viz Frames Stage {py:class}`~morpheus.stages.postprocess.generate_viz_frames_stage.GenerateVizFramesStage` Write out visualization DataFrames.
- MLflow Drift Stage {py:class}`~morpheus.stages.postprocess.ml_flow_drift_stage.MLFlowDriftStage` Report model drift statistics to MLflow.
- Serialize Stage {py:class}`~morpheus.stages.postprocess.serialize_stage.SerializeStage` Include & exclude columns from messages.
- Time Series Stage {py:class}`~morpheus.stages.postprocess.timeseries_stage.TimeSeriesStage` Perform time series anomaly detection and add prediction.

## Pre-process

- Deserialize Stage {py:class}`~morpheus.stages.preprocess.deserialize_stage.DeserializeStage` Partition messages based on the `pipeline_batch_size` parameter of the pipeline's `morpheus.config.Config` object.
- Drop Null Stage {py:class}`~morpheus.stages.preprocess.drop_null_stage.DropNullStage` Drop null data entries from a DataFrame.
- Preprocess FIL Stage {py:class}`~morpheus.stages.preprocess.preprocess_fil_stage.PreprocessFILStage` Prepare FIL input DataFrames for inference.
- Preprocess NLP Stage {py:class}`~morpheus.stages.preprocess.preprocess_nlp_stage.PreprocessNLPStage` Prepare NLP input DataFrames for inference.
