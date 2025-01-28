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
limitations under the Licensesages.cpp
-->

# Morpheus Control Messages

## Background

Control Messages, introduced in version 23.03, provide a solution for numerous use cases that were previously unattainable. This new paradigm enhances the capabilities of Morpheus pipelines by enabling more reactive, event-driven operations. Control Messages involve sending message objects to a pipeline, which can represent a wide range of concepts, from raw data to explicit directives for loading data from specified sources or initiating out-of-band inference or training tasks. The pipeline's behavior can dynamically adapt based on the design; some stages may disregard messages they are not intended to process, while others act according to the message type and content.

This approach unlocks various new applications for Morpheus pipelines. For instance, Control Messages can facilitate real-time data processing and analysis, allowing pipelines to respond promptly to time-sensitive events or data streams. Additionally, they can support adaptive machine learning models that continuously update and refine their predictions based on incoming data. Furthermore, Control Messages can improve resource allocation and efficiency by enabling on-demand data processing and task execution. Overall, the introduction of Control Messages in Morpheus pipelines paves the way for more versatile and responsive software solutions, catering to a broader range of requirements and use cases.

## Anatomy of a Control Message

Control Messages are straightforward objects that contain `tasks`, `metadata`, and possibly `payload` data. Tasks can be one of the following: `training`, `inference`, or `other`. Metadata is a dictionary of key-value pairs that provide additional information about the message and must be JSON serializable. Payload is a Morpheus `MessageMeta` object that can be used to move raw data. Each of these elements can be accessed via the API as the message flows through the pipeline.

### Working with Tasks

Control Messages can handle tasks such as `training`, `inference`, and a catchall category `other`. Tasks can be added, checked for existence, or removed from the Control Message using methods like `add_task`, `has_task`, and `remove_task`.

```python
from morpheus.messages import ControlMessage

task_data = {
    "....": "...."
}

msg = ControlMessage()
msg.add_task("training", task_data)
if msg.has_task("training"):
    task = msg.remove_task("training")
```

### Managing Metadata

Metadata is a set of key-value pairs that offer supplementary information about the Control Message and must be JSON serializable. You can set, check, and retrieve metadata values using the `set_metadata`, `has_metadata`, and `get_metadata` methods, respectively.

```python
from morpheus.messages import ControlMessage

msg = ControlMessage()
msg.set_metadata("description", "This is a sample control message.")
if msg.has_metadata("description"):
    description = msg.get_metadata("description")
```

### Handling Payloads

The payload of a Control Message is a Morpheus `MessageMeta` object that can carry raw data. You can set or retrieve the payload using the `payload` method, which can accept a `MessageMeta` instance or return the payload itself.

```python
import cudf
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta

data = cudf.DataFrame()  # some data

msg_meta = MessageMeta(data)
msg = ControlMessage()

msg.payload(msg_meta)

retrieved_payload = msg.payload()

msg_meta == retrieved_payload # True
```

### Conversion from `MultiMessage` to `ControlMessage`

**The `MultiMessage` type was deprecated in 24.06 and has been completely removed in version 24.10.**

When upgrading to 24.10, all uses of `MultiMessage` need to be converted to `ControlMessage`. Each `MultiMessage` functionality has a corresponding equivalent in `ControlMessage`, as illustrated below.

| **Functionality**                                              | **MultiMessage**                           | **ControlMessage**                                                  |
| -------------------------------------------------------------- | ------------------------------------------ | ------------------------------------------------------------------- |
| Initialization                                                 | `multi_msg = MultiMessage(msg_meta)`       | `control_msg = ControlMessage()`<br>`control_msg.payload(msg_meta)` |
| Get `DataFrame`                                           | `multi_msg.get_meta()`                     | `control_msg.payload().get_data()`                                  |
| Get columns from `DataFrame`                              | `multi_msg.get_meta(col_name)`             | `control_msg.payload().get_data(col_name)`                          |
| Set columns values to `DataFrame`                         | `multi_msg.set_meta(col_name, value)`      | `control_msg.payload().set_data(col_name, value)`                   |
| Get sliced `DataFrame` for given start and stop positions | `multi_msg.get_slice(start, stop)`         | `control_msg.payload().get_slice(start, stop)`                      |
| Copy the `DataFrame` for given ranges of rows             | `multi_msg.copy_ranges(ranges)`            | `control_msg.payload().copy_ranges(ranges)`                         |
|                                                                | **MultiTensorMessage**                     | **ControlMessage**                                                  |
| Get the inference tensor `ndarray`                        | `multi_tensor_msg.tensor()`                | `control_msg.tensors()`                                              |
| Get a specific inference tensor                                 | `multi_tensor_msg.get_tensor(tensor_name)` | `control_msg.tensors().get_tensor(tensor_name)`                                   |


Note that in the `ControlMessage` column the `get_slice()` and `copy_ranges()` methods are being called on the `MessageMeta` payload and thus return a `MessageMeta` after slicing, whereas these functions in `MultiMessage` return a new `MultiMessage` instance.
