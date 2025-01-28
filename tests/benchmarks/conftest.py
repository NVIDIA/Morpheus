# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import asyncio
import glob
import os
import typing
from unittest import mock

import pytest

try:
    from pynvml.smi import NVSMI_QUERY_GPU
    from pynvml.smi import nvidia_smi
except ImportError:
    print("pynvml is not installed")

from test_bench_e2e_pipelines import E2E_TEST_CONFIGS


@pytest.fixture(autouse=True)
def reset_logging_fixture(reset_logging):  # pylint: disable=unused-argument
    yield


# pylint: disable=unused-argument
def pytest_benchmark_update_json(config, benchmarks, output_json):

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
        if bench["name"] not in E2E_TEST_CONFIGS:
            continue

        line_count = 0
        byte_count = 0

        if "file_path" in E2E_TEST_CONFIGS[bench["name"]]:
            source_file = E2E_TEST_CONFIGS[bench["name"]]["file_path"]
            line_count = len(open(source_file, encoding='UTF-8').readlines())  # pylint: disable=consider-using-with
            byte_count = os.path.getsize(source_file)

        elif "input_glob_path" in E2E_TEST_CONFIGS[bench["name"]]:
            for file_name in glob.glob(E2E_TEST_CONFIGS[bench["name"]]["input_glob_path"]):
                line_count += len(open(file_name, encoding='UTF-8').readlines())  # pylint: disable=consider-using-with
                byte_count += os.path.getsize(file_name)

        repeat = E2E_TEST_CONFIGS[bench["name"]]["repeat"]

        bench["morpheus_config"] = {}
        bench["morpheus_config"]["num_threads"] = E2E_TEST_CONFIGS[bench["name"]]["num_threads"]
        bench["morpheus_config"]["pipeline_batch_size"] = E2E_TEST_CONFIGS[bench["name"]]["pipeline_batch_size"]
        bench["morpheus_config"]["model_max_batch_size"] = E2E_TEST_CONFIGS[bench["name"]]["model_max_batch_size"]
        bench["morpheus_config"]["feature_length"] = E2E_TEST_CONFIGS[bench["name"]]["feature_length"]
        bench["morpheus_config"]["edge_buffer_size"] = E2E_TEST_CONFIGS[bench["name"]]["edge_buffer_size"]

        bench['stats']["input_lines"] = line_count * repeat
        bench['stats']['min_throughput_lines'] = (line_count * repeat) / bench['stats']['max']
        bench['stats']['max_throughput_lines'] = (line_count * repeat) / bench['stats']['min']
        bench['stats']['mean_throughput_lines'] = (line_count * repeat) / bench['stats']['mean']
        bench['stats']['median_throughput_lines'] = (line_count * repeat) / bench['stats']['median']
        bench['stats']["input_bytes"] = byte_count * repeat
        bench['stats']['min_throughput_bytes'] = (byte_count * repeat) / bench['stats']['max']
        bench['stats']['max_throughput_bytes'] = (byte_count * repeat) / bench['stats']['min']
        bench['stats']['mean_throughput_bytes'] = (byte_count * repeat) / bench['stats']['mean']
        bench['stats']['median_throughput_bytes'] = (byte_count * repeat) / bench['stats']['median']


@pytest.fixture(name="mock_openai_request_time")
def mock_openai_request_time_fixture():
    return float(os.environ.get("MOCK_OPENAI_REQUEST_TIME", 1.265))


@pytest.fixture(name="mock_nemollm_request_time")
def mock_nemollm_request_time_fixture():
    return float(os.environ.get("MOCK_NEMOLLM_REQUEST_TIME", 0.412))


@pytest.fixture(name="mock_web_scraper_request_time")
def mock_web_scraper_request_time_fixture():
    return float(os.environ.get("MOCK_WEB_SCRAPER_REQUEST_TIME", 0.5))


@pytest.fixture(name="mock_feedparser_request_time")
def mock_feedparser_request_time_fixture():
    return float(os.environ.get("MOCK_FEEDPARSER_REQUEST_TIME", 0.5))


@pytest.fixture(name="mock_serpapi_request_time")
def mock_serpapi_request_time_fixture():
    return float(os.environ.get("MOCK_SERPAPI_REQUEST_TIME", 1.7))


@pytest.mark.usefixtures("openai")
@pytest.fixture(name="mock_chat_completion")
def mock_chat_completion_fixture(mock_chat_completion: tuple[mock.MagicMock, mock.MagicMock],
                                 mock_openai_request_time: float):
    (mock_client, mock_async_client) = mock_chat_completion

    async def sleep_first(*args, **kwargs):
        # Sleep time is based on average request time
        await asyncio.sleep(mock_openai_request_time)
        return mock.DEFAULT

    mock_async_client.chat.completions.create.side_effect = sleep_first
    mock_client.chat.completions.create.side_effect = sleep_first

    yield (mock_client, mock_async_client)


@pytest.mark.usefixtures("nemollm")
@pytest.fixture(name="mock_nemollm")
def mock_nemollm_fixture(mock_nemollm: mock.MagicMock, mock_nemollm_request_time: float):

    # The generate function is a blocking call that returns a future when return_type="async"
    async def sleep_first(fut: asyncio.Future, value: typing.Any = mock.DEFAULT):
        # Sleep time is based on average request time
        await asyncio.sleep(mock_nemollm_request_time)
        fut.set_result(value)

    def create_future(*args, **kwargs) -> asyncio.Future:
        event_loop = asyncio.get_event_loop()
        fut = event_loop.create_future()
        event_loop.create_task(sleep_first(fut, mock.DEFAULT))
        return fut

    mock_nemollm.generate.side_effect = create_future

    yield mock_nemollm
