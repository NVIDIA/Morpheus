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

# Troubleshooting

**Out of Date Build Cache**

The Morpheus build system, by default, stores all build artifacts and cached output in the `${MORPHEUS_ROOT}/.cache` directory. This cache directory is designed to speed up successive builds but occasionally can get out of date and cause unexpected build errors. In this situation, it's best to completely delete the build and cache directories and restart the build:

```bash
# Delete the build and cache folders
rm -rf ${MORPHEUS_ROOT}/.cache
rm -rf ${MORPHEUS_ROOT}/build

# Clean out documentation builds:
rm -rf docs/source/_modules docs/source/_lib

# Clean out shared-libs if compiled with `MORPHEUS_PYTHON_INPLACE_BUILD=ON`:
find ./morpheus -name "*.so" -delete

# Clean out shared libs if examples have been built:
find ./examples -name "*.so" -delete

# Restart the build
./scripts/compile.sh
```

**Incompatible MLflow Models**

Models trained with a previous version of Morpheus and stored into MLflow may be incompatible with the current version. This error can be identified by the following error message occurring in an MLflow based pipeline such as DFP.

```
Error trying to get model

Traceback (most recent call last):

File "/workspace/python/morpheus_dfp/morpheus_dfp/stages/dfp_inference_stage.py", line 101, in on_data

loaded_model = model_cache.load_model(self._client)
```
```
ModuleNotFoundError: No module named 'dfencoder'
```

The workarounds available for this issue are:

* Revert to the previous version of Morpheus until the models can be re-trained.
* Re-train the model using the current version of Morpheus

In the case of models trained by the DFP example, the existing models can be deleted by running the following command:

```bash
docker volume ls # list current docker volumes
docker volume rm production_db_data production_mlflow_data

# Re-build the MLflow container for DFP
cd ${MORPHEUS_ROOT}/examples/digital_fingerprinting/production/
docker compose build
docker compose up mlflow
```


**Debugging Python Code**

To debug issues in python code, several Visual Studio Code launch configurations have been included in the repo. These launch configurations can be found in `${MORPHEUS_ROOT}/morpheus.code-workspace`. To launch the debugging environment, ensure Visual Studio Code has opened the Morpheus workspace file (File->Open Workspace from File...). Once the workspace has been loaded, the launch configurations should be available in the debugging tab.

**Debugging C++ Code**

Similar to the Python launch configurations, several C++ launch configurations can be found in the Visual Studio Code workspace file. However, unlike the Python configuration, it's necessary to ensure Morpheus was compiled in Debug mode in order for breakpoints to work correctly. To build Morpheus in Debug mode, use the following:

```bash
CMAKE_CONFIGURE_EXTRA_ARGS="-DCMAKE_BUILD_TYPE=Debug" ./scripts/compile.sh
```
