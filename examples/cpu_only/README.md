<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# CPU Only Example Using Morpheus

## Supported Environments
| Environment | Supported | Notes |
|-------------|-----------|-------|
| Conda | ✔ | |
| Morpheus Docker Container | ✔ |  |
| Morpheus Release Container | ✔ |  |
| Dev Container | ✔ |  |

## CPU Only Pipeline
This example demonstrates a simple Morpheus pipeline which is able to operate on a host without access GPU.

> **Note**: A more complex example of a pipeline that can execute without a GPU is also available at `examples/llm/completion/README.md`

From the root of the Morpheus repo, run:
```bash
python examples/cpu_only/run.py --help
```

Output:
```
Usage: run.py [OPTIONS]

Options:
  --use_cpu_only                  Whether or not to run in CPU only mode,
                                  setting this to True will disable C++ mode.
  --log_level [CRITICAL|FATAL|ERROR|WARN|WARNING|INFO|DEBUG]
                                  Specify the logging level to use.  [default:
                                  DEBUG]
  --in_file PATH                  Input file  [required]
  --out_file FILE                 Output file  [required]
  --help                          Show this message and exit.
```

To launch the configured Morpheus pipeline with the sample data that is provided in `examples/data`, run the following:

```bash
python examples/cpu_only/run.py --use_cpu_only --in_file=examples/data/email.jsonlines --out_file=.tmp/output/cpu_only_out.jsonlines
```

### CLI Example

From the root of the Morpheus repo, run:
```bash
morpheus --log_level INFO \
    run --use_cpu_only \
    pipeline-other \
    from-file --filename=examples/data/email.jsonlines \
    monitor --description "source" \
    deserialize \
    monitor --description "deserialize" \
    serialize \
    to-file --filename=.tmp/output/cpu_only_cli_out.jsonlines --overwrite
```
