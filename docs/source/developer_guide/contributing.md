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

# Contributing to Morpheus

Contributions to Morpheus fall into the following three categories.

1. To report a bug, request a new feature, or report a problem with
    documentation, file an [issue](https://github.com/nv-morpheus/Morpheus/issues/new/choose)
    describing in detail the problem or new feature. The Morpheus team evaluates
    and triages issues, and schedules them for a release. If you believe the
    issue needs priority attention, comment on the issue to notify the
    team.
2. To propose and implement a new Feature, file a new feature request
    [issue](https://github.com/nv-morpheus/Morpheus/issues/new/choose). Describe the
    intended feature and discuss the design and implementation with the team and
    community. Once the team agrees that the plan is good, go ahead and
    implement it, using the [code contributions](#code-contributions) guide below.
3. To implement a feature or bug-fix for an existing outstanding issue,
    follow the [code contributions](#code-contributions) guide below. If you
    need more context on a particular issue, ask in a comment.

As contributors and maintainers to this project,
you are expected to abide by Morpheus' code of conduct.
More information can be found at: [Contributor Code of Conduct](https://github.com/nv-morpheus/.github/blob/main/CODE_OF_CONDUCT.md).

## Code contributions

### Your first issue

1. Find an issue to work on. The best way is to search for issues with the [good first issue](https://github.com/nv-morpheus/Morpheus/issues) label.
2. Comment on the issue stating that you are going to work on it.
3. Code! Make sure to update unit tests! Ensure the [license headers are set properly](#licensing).
4. When done, [create your pull request](https://github.com/nv-morpheus/Morpheus/compare).
5. Wait for other developers to review your code and update code as needed.
6. Once reviewed and approved, a Morpheus developer will merge your pull request.

Remember, if you are unsure about anything, don't hesitate to comment on issues and ask for clarifications!

### Seasoned developers

Once you have gotten your feet wet and are more comfortable with the code, you can review the prioritized issues for our next release in our [project boards](https://github.com/nv-morpheus/Morpheus/projects).

> **Pro Tip:** Always review the release board with the highest number for issues to work on. This is where Morpheus developers also focus their efforts.

Review the unassigned issues, and find an issue to which you are comfortable contributing. Start with _Step 2_ above, commenting on the issue to let others know you are working on it. If you have any questions related to the implementation of the issue, ask them in the issue instead of the PR.

## Building from Source

The following instructions are for developers who are getting started with the Morpheus repository. The Morpheus development environment is flexible (Docker, Conda and bare metal workflows) but has a high number of dependencies that can be difficult to set up. These instructions outline the steps for setting up a development environment inside a Docker container or on a host machine with Conda.

All of the following instructions assume that the given variable has been defined:
 - `MORPHEUS_ROOT`: The Morpheus repository has been checked out at a location specified by this variable. Any non-absolute paths are relative to `MORPHEUS_ROOT`.


### Clone the repository and pull large file data from Git LFS

```bash
export MORPHEUS_ROOT=$(pwd)/morpheus
git clone https://github.com/nv-morpheus/Morpheus.git $MORPHEUS_ROOT
cd $MORPHEUS_ROOT
```

Ensure all submodules are checked out:

```bash
git submodule update --init --recursive
```

The large model and data files in this repo are stored using [Git Large File Storage (LFS)](https://git-lfs.github.com/). These files will be required for running the training/validation scripts and example pipelines for the Morpheus pre-trained models.

By default only those files stored in LFS strictly needed for running Morpheus are included when the Morpheus repository is cloned. Additional datasets can be downloaded using the `scripts/fetch_data.py` script. Refer to the section [Git LFS](../getting_started.md#git-lfs) of the [getting_started.md](../getting_started.md) guide for details on this.

### Build in Docker Container

This workflow utilizes a Docker container to set up most dependencies ensuring a consistent environment.

#### Prerequisites

1. Ensure all [requirements](../getting_started.md#requirements) from [getting_started.md](../getting_started.md) are met.
1. Build the development container
   ```bash
   ./docker/build_container_dev.sh
   ```
   1. The container tag will default to `morpheus:YYMMDD` where `YYMMDD` is the current 2 digit year, month and day respectively. The tag can be overridden by setting `DOCKER_IMAGE_TAG`. For example,
      ```bash
      DOCKER_IMAGE_TAG=my_tag ./docker/build_container_dev.sh
      ```
      Would build the container `morpheus:my_tag`.
   1. To build the container with a debugging version of CPython installed, update the Docker target as follows:
   ```shell
   DOCKER_TARGET=development_pydbg ./docker/build_container_dev.sh
   ```
   1. Note: When debugging Python code, you just need to add `ci/conda/recipes/python-dbg/source` to the source path your debugger.
   1. Once created, you will be able to introspect Python objects from within GDB. For example, if we were to break within a generator setup call and examine its `PyFrame_Object` `f`, it might be similar to:
   ```shell
    #4  0x000056498ce685f4 in gen_send_ex (gen=0x7f3ecc07ad40, arg=<optimized out>, exc=<optimized out>, closing=<optimized out>) at Objects/genobject.c:222
    (gdb) pyo f
    object address  : 0x7f3eb3888750
    object refcount : 1
    object type     : 0x56498cf99c00
    object type name: frame
    object repr     : <frame at 0x7f3eb3888750, file '/workspace/morpheus/pipeline/pipeline.py', line 902, code join
   ```
   1. Note: Now when running the container, Conda should list your Python version as `pyxxx_dbg_morpheus`.
   ```shell
    (morpheus) user@host:/workspace# conda list | grep python
    python                    3.8.13          py3.8.13_dbg_morpheus    local
   ```
   1. Note: This does not build any Morpheus or MRC code and defers building the code until the entire repo can be mounted into a running container. This allows for faster incremental builds during development.
2. Run the development container
   ```bash
   ./docker/run_container_dev.sh
   ```
   1. The container tag follows the same rules as `build_container_dev.sh` and will default to the current `YYMMDD`. Specify the desired tag with `DOCKER_IMAGE_TAG`. For example, `DOCKER_IMAGE_TAG=my_tag ./docker/run_container_dev.sh`
   2. This will automatically mount the current working directory to `/workspace`.
   3. Some of the validation tests require launching the Morpheus models Docker container within the Morpheus container. To enable this you will need to grant the Morpheus container access to your host OS's Docker socket file with:
      ```bash
      DOCKER_EXTRA_ARGS="-v /var/run/docker.sock:/var/run/docker.sock" ./docker/run_container_dev.sh
      ```
      Then once the container is started you will need to install some extra packages to enable launching Docker containers:
      ```bash
      ./external/utilities/docker/install_docker.sh
      ```

3. Compile Morpheus
   ```bash
   ./scripts/compile.sh
   ```
   This script will run CMake Configure with default options, the CMake build and install Morpheus into the environment.

4. [Run Morpheus](../getting_started.md#running-morpheus)
   ```bash
   morpheus run pipeline-nlp ...
   ```
   At this point, Morpheus can be fully used. Any changes to Python code will not require a rebuild. Changes to C++ code will require calling `./scripts/compile.sh`.

### Build in a Conda Environment

If a [Conda](https://docs.conda.io/projects/conda/en/latest/) environment on the host machine is preferred over Docker, it is relatively easy to install the necessary dependencies (In reality, the Docker workflow creates a Conda environment inside the container).

#### Conda Environment YAML Files
Morpheus provides multiple Conda environment files to support different workflows. Morpheus utilizes [rapids-dependency-file-generator](https://pypi.org/project/rapids-dependency-file-generator/) to manage these multiple environment files. All of Morpheus' Conda and [pip](https://pip.pypa.io/en/stable/) dependencies along with the different environments are defined in the `dependencies.yaml` file.

The following are the available Conda environment files, all are located in the `conda/environments` directory, with the following naming convention: `<environment>_<cuda_version>_arch-<architecture>.yaml`.
| Environment | File | Description |
| --- | --- | --- |
| `all` | `all_cuda-125_arch-<arch>.yaml` | All dependencies required to build, run and test Morpheus, along with all of the examples. This is a superset of the `dev`, `runtime` and `examples` environments. |
| `dev` | `dev_cuda-125_arch-<arch>.yaml` | Dependencies required to build, run and test Morpheus. This is a superset of the `runtime` environment. |
| `examples` | `examples_cuda-125_arch-<arch>.yaml` | Dependencies required to run all examples. This is a superset of the `runtime` environment. |
| `model-utils` | `model-utils_cuda-125_arch-<arch>.yaml` | Dependencies required to train models independent of Morpheus. |
| `runtime` | `runtime_cuda-125_arch-<arch>.yaml` | Minimal set of dependencies strictly required to run Morpheus. |


##### Updating Morpheus Dependencies
Changes to Morpheus dependencies can be made in the `dependencies.yaml` file, then run `rapids-dependency-file-generator` to update the individual environment files in the `conda/environments` directory  .

Install `rapids-dependency-file-generator` into the base Conda environment:
```bash
conda run -n base --live-stream pip install rapids-dependency-file-generator
```

Then to generate update the individual environment files run:
```bash
conda run -n base --live-stream rapids-dependency-file-generator
```

When ready, commit both the changes to the `dependencies.yaml` file and the updated environment files into the repo.

#### Prerequisites

- Volta architecture GPU or better
- [CUDA 12.5](https://developer.nvidia.com/cuda-12-5-0-download-archive)
- `conda`
  - If `conda` is not installed, we recommend using the [MiniForge install guide](https://github.com/conda-forge/miniforge). This will install `conda` and set the channel default to use `conda-forge`.

1. Set up environment variables and clone the repo:
   ```bash
   export MORPHEUS_ROOT=$(pwd)/morpheus
   git clone https://github.com/nv-morpheus/Morpheus.git $MORPHEUS_ROOT
   cd $MORPHEUS_ROOT
   ```
1. Ensure all submodules are checked out:
   ```bash
   git submodule update --init --recursive
   ```
1. Create the Morpheus Conda environment using either the `dev` or `all` environment file. Refer to the [Conda Environment YAML Files](#conda-environment-yaml-files) section for more information.
   ```bash
   conda env create --solver=libmamba -n morpheus --file conda/environments/dev_cuda-125_arch-$(arch).yaml
   ```
   or
   ```bash
   conda env create --solver=libmamba -n morpheus --file conda/environments/all_cuda-125_arch-$(arch).yaml

   ```

   This creates a new environment named `morpheus`. Activate the environment with:
   ```bash
   conda activate morpheus
   ```

1. Build Morpheus
   ```bash
   ./scripts/compile.sh
   ```
   This script will build and install Morpheus into the Conda environment.
1. Test the build (Note: some tests will be skipped)
   Some of the tests will rely on external data sets.
   ```bash
   MORPHEUS_ROOT=${PWD}

   git lfs install
   git lfs update
   ./scripts/fetch_data.py fetch all
   ```
   This script will fetch the data sets needed. Then run:
   ```bash
   pytest
   ```
1. Optional: Run full end-to-end tests
   - Our end-to-end tests require the [camouflage](https://testinggospels.github.io/camouflage/) testing framework. Install camouflage with:
      ```bash
      npm install -g camouflage-server@0.15
      ```

   - Run end-to-end (aka slow) tests:
      ```bash
      pytest --run_slow
      ```
1. Optional: Run Kafka and Milvus tests
   - Download Kafka:
      ```bash
      python ./ci/scripts/download_kafka.py
      ```

   - Run all tests (this will skip over tests that require optional dependencies which are not installed):
      ```bash
      pytest --run_slow --run_kafka --run_milvus
      ```

   - Run all tests including those that require optional dependencies:
      ```bash
      pytest --fail_missing --run_slow --run_kafka --run_milvus
      ```

1. Run Morpheus
   ```bash
   morpheus run pipeline-nlp ...
   ```
   At this point, Morpheus can be fully used. Any changes to Python code will not require a rebuild. Changes to C++ code will require calling `./scripts/compile.sh`. Installing Morpheus is only required once per virtual environment.

### Build the Morpheus Models Container

From the root of the Morpheus repository run the following command:
```bash
models/docker/build_container.sh
```

### Quick Launch Kafka Cluster

Launching a full production Kafka cluster is outside the scope of this project; however, if a quick cluster is needed for testing or development, one can be quickly launched via Docker Compose. The following commands outline that process. Refer to [this](https://medium.com/big-data-engineering/hello-kafka-world-the-complete-guide-to-kafka-with-docker-and-python-f788e2588cfc) guide for more in-depth information:

1. Install `docker-compose-plugin` if not already installed:

   ```bash
   apt-get update
   apt-get install docker-compose-plugin
   ```
2. Clone the `kafka-docker` repo from the Morpheus repo root:

   ```bash
   git clone https://github.com/wurstmeister/kafka-docker.git
   ```
3. Change directory to `kafka-docker`:

   ```bash
   cd kafka-docker
   ```
4. Export the IP address of your Docker `bridge` network:

   ```bash
   export KAFKA_ADVERTISED_HOST_NAME=$(docker network inspect bridge | jq -r '.[0].IPAM.Config[0].Gateway')
   ```
5. Update the `kafka-docker/docker-compose.yml`, performing two changes:
   1. Update the `ports` entry to:
      ```yaml
      ports:
         - "0.0.0.0::9092"
      ```
      This will prevent the containers from attempting to map IPv6 ports.
   1. Change the value of `KAFKA_ADVERTISED_HOST_NAME` to match the value of the `KAFKA_ADVERTISED_HOST_NAME` environment variable from the previous step. For example, the line should be similar to:

      ```yaml
      environment:
         KAFKA_ADVERTISED_HOST_NAME: 172.17.0.1
      ```
      Which should match the value of `$KAFKA_ADVERTISED_HOST_NAME` from the previous step:

      ```bash
      $ echo $KAFKA_ADVERTISED_HOST_NAME
      "172.17.0.1"
      ```
6. Launch Kafka with 3 instances:

   ```bash
   docker compose up -d --scale kafka=3
   ```
   In practice, 3 instances have been shown to work well. Use as many instances as required. Keep in mind each instance takes about 1 GB of memory.
7. Launch the Kafka shell
   1. To configure the cluster, you will need to launch into a container that has the Kafka shell.
   2. You can do this with:
      ```bash
      ./start-kafka-shell.sh $KAFKA_ADVERTISED_HOST_NAME
      ```
   3. However, this makes it difficult to load data into the cluster. Instead, you can manually launch the Kafka shell by running:
      ```bash
      # Change to the morpheus root to make it easier for mounting volumes
      cd ${MORPHEUS_ROOT}

      # Run the Kafka shell Docker container
      docker run --rm -it -v /var/run/docker.sock:/var/run/docker.sock \
         -e HOST_IP=$KAFKA_ADVERTISED_HOST_NAME -e ZK=$2 \
         -v $PWD:/workspace wurstmeister/kafka /bin/bash
      ```
      Note the `-v $PWD:/workspace`. This will make anything in your current directory available in `/workspace`.
   4. Once the Kafka shell has been launched, you can begin configuring the cluster. All of the following commands require the argument `--bootstrap-server`. To simplify things, set the `BOOTSTRAP_SERVER` and `MY_TOPIC` variables:
      ```bash
      export BOOTSTRAP_SERVER=$(broker-list.sh)
      export MY_TOPIC="your_topic_here"
      ```
8. Create the topic

   ```bash
   # Create the topic
   kafka-topics.sh --bootstrap-server ${BOOTSTRAP_SERVER} --create --topic ${MY_TOPIC}

   # Change the number of partitions
   kafka-topics.sh --bootstrap-server ${BOOTSTRAP_SERVER} --alter --topic ${MY_TOPIC} --partitions 3

   # Refer to the topic info
   kafka-topics.sh --bootstrap-server ${BOOTSTRAP_SERVER} --describe --topic=${MY_TOPIC}
   ```
   **Note:** If you are using `to-kafka`, ensure your output topic is also created.

9. Generate input messages
   1. In order for Morpheus to read from Kafka, messages need to be published to the cluster. You can use the `kafka-console-producer.sh` script to load data:

      ```bash
      kafka-console-producer.sh --bootstrap-server ${BOOTSTRAP_SERVER} --topic ${MY_TOPIC} < ${FILE_TO_LOAD}
      ```

      **Note:** In order for this to work, your input file must be accessible from the current directory the Kafka shell was launched from.

   2. You can view the messages with:

      ```bash
      kafka-console-consumer.sh --bootstrap-server ${BOOTSTRAP_SERVER} --topic ${MY_TOPIC}
      ```

      **Note:** This will consume messages.

### Pipeline Validation

To verify that all pipelines are working correctly, validation scripts have been added at `${MORPHEUS_ROOT}/scripts/validation`. There are scripts for each of the main workflows: Anomalous Behavior Profiling (ABP), Phishing Detection (Phishing), and Sensitive Information Detection (SID).

To run all of the validation workflow scripts, use the following commands:

```bash
# Run validation scripts
./scripts/validation/val-run-all.sh
```

At the end of each workflow, a section will print the different inference workloads that were run and the validation error percentage for each. For example:

```bash
===ERRORS===
PyTorch     :3/314 (0.96 %)
Triton(ONNX):Skipped
Triton(TRT) :Skipped
TensorRT    :Skipped
Complete!
```

This indicates that only 3 out of 314 rows did not match the validation dataset. Errors similar to `:/ ( %)` or very high percentages display, then the workflow did not complete successfully.

### Troubleshooting the Build

Due to the large number of dependencies, it's common to run into build issues. The follow are some common issues, tips, and suggestions:

 - Issues with the build cache
   - To avoid rebuilding every compilation unit for all dependencies after each change, a fair amount of the build is cached. By default, the cache is located at `${MORPHEUS_ROOT}/.cache`. The cache contains both compiled object files, source repositories, ccache files, clangd files and even the cuDF build.
   - The entire cache folder can be deleted at any time and will be redownload/recreated on the next build
 - Message indicating `git apply ...` failed
   - Many of the dependencies require small patches to make them work. These patches must be applied once and only once. If this error displays, try deleting the offending package from the `build/_deps/<offending_package>` directory or from `.cache/cpm/<offending_package>`.
   - If all else fails, delete the entire `build/` directory and `.cache/` directory.
 - Older build artifacts when performing an in-place build.
   - When built with `MORPHEUS_PYTHON_INPLACE_BUILD=ON` compiled libraries will be deployed in-place in the source tree, and older build artifacts exist in the source tree. Remove these with:
       ```bash
       find ./python -name "*.so" -delete
       find ./examples -name "*.so" -delete
       ```
 - Issues building documentation
   - Intermediate documentation build artifacts can cause errors for Sphinx. To remove these, run:
       ```bash
       rm -rf build/docs/ docs/source/_modules docs/source/_lib
       ```
 - CI Issues
   - To run CI locally, the `ci/scripts/run_ci_local.sh` script can be used. For example to run a local CI build:
      ```bash
      ci/scripts/run_ci_local.sh build
      ```
      - Build artifacts resulting from a local CI run can be found in the `.tmp/local_ci_tmp/` directory.
   - To troubleshoot a particular CI stage it can be helpful to run:
      ```bash
      ci/scripts/run_ci_local.sh bash
      ```

      This will open a bash shell inside the CI container with all of the environment variables typically set during a CI run. From here you can run the commands that would typically be run by one of the CI scripts in `ci/scripts/github`.

      To run a CI stage requiring a GPU (ex: `test`), set the `USE_GPU` environment variable to `1`:
      ```bash
      USE_GPU=1 ci/scripts/run_ci_local.sh bash
      ```

Refer to the [troubleshooting guide](../extra_info/troubleshooting.md) for more information on common issues and how to resolve them.

## Licensing
Morpheus is licensed under the Apache v2.0 license. All new source files including CMake and other build scripts should contain the Apache v2.0 license header. Any edits to existing source code should update the date range of the copyright to the current year. The format for the license header is:

```
/*
 * SPDX-FileCopyrightText: Copyright (c) <year>, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
 ```

### Third-party code
Third-party code included in the source tree (that is not pulled in as an external dependency) must be compatible with the Apache v2.0 license and should retain the original license along with a URL to the source. If this code is modified, it should contain both the Apache v2.0 license followed by the original license of the code and the URL to the original code.

Ex:
```
/**
 * SPDX-FileCopyrightText: Copyright (c) <year>, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//
// Original Source: https://github.com/org/other_project
//
// Original License:
// ...
```


---

## Attribution
Portions adopted from
* [https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md](https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md)
* [https://github.com/dask/dask/blob/master/docs/source/develop.rst](https://github.com/dask/dask/blob/master/docs/source/develop.rst)
