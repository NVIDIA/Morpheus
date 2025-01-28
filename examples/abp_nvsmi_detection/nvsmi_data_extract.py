# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import time

import pandas as pd
from pynvml.smi import NVSMI_QUERY_GPU
from pynvml.smi import nvidia_smi


def main(args):
    query_opts = NVSMI_QUERY_GPU.copy()

    # Remove the timestamp and supported clocks from the query
    del query_opts["timestamp"]
    del query_opts["supported-clocks"]

    nvsmi = nvidia_smi.getInstance()

    with open(args.output_file, "w", encoding="UTF-8") as f:

        while (True):

            device_query = nvsmi.DeviceQuery(list(query_opts.values()))

            output_dicts = []

            # Flatten the GPUs to allow for a new row per GPU
            for gpu in device_query["gpu"]:
                single_gpu = device_query.copy()

                # overwrite the gpu list with a single gpu
                single_gpu["gpu"] = gpu

                output_dicts.append(single_gpu)

            df = pd.json_normalize(output_dicts, record_prefix="nvidia_smi_log")

            # Rename the id column to match the XML converted output from NetQ
            df.rename(columns={"gpu.id": "gpu.@id", "count": "attached_gpus"}, inplace=True)

            df.rename(columns=lambda x: "nvidia_smi_log" + "." + x, inplace=True)

            # Add the current timestamp
            df.insert(0, "timestamp", time.time())

            df.to_json(f, orient="records", lines=True)

            f.flush()

            time.sleep(args.interval_ms / 1000.0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interval-ms', default=1000, help='interval in ms between writes to output file')
    parser.add_argument("--output-file", default='nvsmi.jsonlines', help='output file to save dataset')

    main(parser.parse_args())
