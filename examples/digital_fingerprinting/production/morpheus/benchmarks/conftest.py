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

import glob
import json
from os import path

from pynvml.smi import NVSMI_QUERY_GPU
from pynvml.smi import nvidia_smi

from benchmarks.test_bench_e2e_dfp_pipeline import PIPELINES_CONF


def get_json_lines_count(filename):
    with open(filename, encoding='utf-8') as f:
        lines = json.loads(f.read())
    return len(lines)


def pytest_benchmark_update_json(config, benchmarks, output_json):  # pylint:disable=unused-argument

    curr_dir = path.dirname(path.abspath(__file__))

    query_opts = NVSMI_QUERY_GPU.copy()
    nvsmi = nvidia_smi.getInstance()
    device_query = nvsmi.DeviceQuery([
        query_opts["driver_version"],
        query_opts["count"],
        query_opts["index"],
        query_opts["gpu_name"],
        query_opts["gpu_uuid"],
        query_opts["memory.total"],
        query_opts["memory.used"],
        query_opts["memory.free"],
        query_opts["utilization.gpu"],
        query_opts["utilization.memory"],
        query_opts["temperature.gpu"]
    ])

    output_json["machine_info"]["gpu_driver_version"] = device_query["driver_version"]

    for gpu in device_query["gpu"]:
        gpu_num = gpu["minor_number"]
        output_json["machine_info"]["gpu_" + gpu_num] = {}
        output_json["machine_info"]["gpu_" + gpu_num]["id"] = gpu_num
        output_json["machine_info"]["gpu_" + gpu_num]["name"] = gpu["product_name"]
        output_json["machine_info"][
            "gpu_" + gpu_num]["utilization"] = f"{gpu['utilization']['gpu_util']}{gpu['utilization']['unit']}"
        output_json["machine_info"][
            "gpu_" + gpu_num]["total_memory"] = f"{gpu['fb_memory_usage']['total']} {gpu['fb_memory_usage']['unit']}"
        output_json["machine_info"][
            "gpu_" + gpu_num]["used_memory"] = f"{gpu['fb_memory_usage']['used']} {gpu['fb_memory_usage']['unit']}"
        output_json["machine_info"][
            "gpu_" + gpu_num]["free_memory"] = f"{gpu['fb_memory_usage']['free']} {gpu['fb_memory_usage']['unit']}"
        output_json["machine_info"][
            "gpu_" + gpu_num]["temperature"] = f"{gpu['temperature']['gpu_temp']} {gpu['temperature']['unit']}"
        output_json["machine_info"]["gpu_" + gpu_num]["uuid"] = gpu["uuid"]

    for bench in output_json['benchmarks']:

        line_count = 0
        byte_count = 0

        if "file_path" in PIPELINES_CONF[bench["name"]]:
            source_file = PIPELINES_CONF[bench["name"]]["file_path"]
            source_file = path.join(curr_dir, source_file)
            line_count = get_json_lines_count(source_file)
            byte_count = path.getsize(source_file)

        elif "glob_path" in PIPELINES_CONF[bench["name"]]:
            source_files_glob = path.join(curr_dir, PIPELINES_CONF[bench["name"]]["glob_path"])
            for filename in glob.glob(source_files_glob):
                line_count += get_json_lines_count(filename)
                byte_count += path.getsize(filename)
        elif "message_path" in PIPELINES_CONF[bench["name"]]:
            source_message_glob = path.join(curr_dir, PIPELINES_CONF[bench["name"]]["message_path"])
            for message_fn in glob.glob(source_message_glob):
                with open(message_fn, encoding='utf-8') as message_file:
                    control_message = json.load(message_file)
                inputs = control_message.get("inputs")
                # Iterating over inputs
                for inpt in inputs:
                    non_load_task = None
                    line_count_per_task = 0
                    byte_count_per_task = 0
                    tasks = inpt.get("tasks")
                    # Iterating over tasks
                    for task in tasks:
                        if task.get("type") == "load":
                            files = task.get("properties").get("files")
                            # Iterating over files in a task
                            for file_glob in files:
                                # Iterating over a file glob
                                for file_name in glob.glob(file_glob):
                                    count = get_json_lines_count(file_name)
                                    size = path.getsize(file_name)
                                    line_count += count
                                    byte_count += size
                                    line_count_per_task += count
                                    byte_count_per_task += size
                        else:
                            non_load_task = task.get("type")
                    # Adding non-load task status here.
                    if non_load_task is not None:
                        bench['stats'][non_load_task] = {}
                        bench['stats'][non_load_task]["input_lines"] = line_count_per_task
                        bench['stats'][non_load_task]["input_bytes"] = byte_count_per_task

        else:
            raise KeyError("Configuration requires either 'glob_path' or 'file_path' attribute.")

        bench["morpheus_config"] = {}
        bench["morpheus_config"]["num_threads"] = PIPELINES_CONF[bench["name"]]["num_threads"]
        bench["morpheus_config"]["pipeline_batch_size"] = PIPELINES_CONF[bench["name"]]["pipeline_batch_size"]
        bench["morpheus_config"]["edge_buffer_size"] = PIPELINES_CONF[bench["name"]]["edge_buffer_size"]
        bench["morpheus_config"]["start_time"] = PIPELINES_CONF[bench["name"]]["start_time"]
        bench["morpheus_config"]["duration"] = PIPELINES_CONF[bench["name"]]["duration"]

        bench['stats']["input_lines"] = line_count
        bench['stats']['min_throughput_lines'] = line_count / bench['stats']['max']
        bench['stats']['max_throughput_lines'] = line_count / bench['stats']['min']
        bench['stats']['mean_throughput_lines'] = line_count / bench['stats']['mean']
        bench['stats']['median_throughput_lines'] = line_count / bench['stats']['median']
        bench['stats']["input_bytes"] = byte_count
        bench['stats']['min_throughput_bytes'] = byte_count / bench['stats']['max']
        bench['stats']['max_throughput_bytes'] = byte_count / bench['stats']['min']
        bench['stats']['mean_throughput_bytes'] = byte_count / bench['stats']['mean']
        bench['stats']['median_throughput_bytes'] = byte_count / bench['stats']['median']
