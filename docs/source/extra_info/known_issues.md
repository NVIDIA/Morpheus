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

# Known Issues

- `vdb_upload` example pipeline triggers an internal error in Triton ([#1649](https://github.com/nv-morpheus/Morpheus/issues/1649))
- `ransomware_detection` example pipeline occasionally logs a `distributed.comm.core.CommClosedError` error during shutdown ([#2026](https://github.com/nv-morpheus/Morpheus/issues/2026)).
- `abp_pcap_detection` pipeline running slowly on AArch64 ([#2120](https://github.com/nv-morpheus/Morpheus/issues/2120))
- LLM `vdb_upload` and `rag` pipelines not supported on AArch64 ([#2122](https://github.com/nv-morpheus/Morpheus/issues/2122))
- `gnn_fraud_detection_pipeline` not working on AArch64 ([#2123](https://github.com/nv-morpheus/Morpheus/issues/2123))
- `ransomware_detection` pipeline running slowly on AArch64 ([#2124](https://github.com/nv-morpheus/Morpheus/issues/2124))
- DFP visualization fails to install on AArch64 ([#2125](https://github.com/nv-morpheus/Morpheus/issues/2125))

Refer to [open issues in the Morpheus project](https://github.com/nv-morpheus/Morpheus/issues)
