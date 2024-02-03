# Copyright (c) 2023-2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time

import cudf

from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.io.deserializers import read_file_to_df
from morpheus.llm import LLMEngine
from morpheus.llm.nodes.extracter_node import ExtracterNode
from morpheus.llm.nodes.llm_generate_node import LLMGenerateNode
from morpheus.llm.nodes.prompt_template_node import PromptTemplateNode
from morpheus.llm.services.llm_service import LLMService
from morpheus.llm.services.nemo_llm_service import NeMoLLMService
from morpheus.llm.services.openai_chat_service import OpenAIChatService
from morpheus.llm.task_handlers.simple_task_handler import SimpleTaskHandler
from morpheus.messages import ControlMessage
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.llm.llm_engine_stage import LLMEngineStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.utils.concat_df import concat_dataframes

logger = logging.getLogger(__name__)


def _build_engine(llm_service: str):

    llm_service_cls: type[LLMService] = None
    model_name: str = None

    if llm_service == "NemoLLM":
        llm_service_cls = NeMoLLMService
        model_name = "gpt-43b-002"
    elif llm_service == "OpenAI":
        llm_service_cls = OpenAIChatService
        model_name = 'gpt-3.5-turbo'
    else:
        raise ValueError(f"Invalid LLM service: {llm_service}")

    llm_service = llm_service_cls()
    llm_client = llm_service.get_client(model_name=model_name)

    engine = LLMEngine()

    engine.add_node("extracter", node=ExtracterNode())

    engine.add_node("prompts",
                    inputs=["/extracter"],
                    node=PromptTemplateNode(template="What is the capital of {{country}}?", template_format="jinja"))

    engine.add_node("completion", inputs=["/prompts"], node=LLMGenerateNode(llm_client=llm_client))

    engine.add_task_handler(inputs=["/completion"], handler=SimpleTaskHandler())

    return engine


def pipeline(num_threads: int,
             pipeline_batch_size: int,
             model_max_batch_size: int,
             repeat_count: int,
             llm_service: str,
             input_file: str = None,
             shuffle: bool = False) -> float:

    config = Config()

    # Below properties are specified by the command line
    config.num_threads = num_threads
    config.pipeline_batch_size = pipeline_batch_size
    config.model_max_batch_size = model_max_batch_size
    config.mode = PipelineModes.NLP
    config.edge_buffer_size = 128

    if input_file is not None:
        source_df = read_file_to_df(input_file, df_type='cudf')
    else:
        source_df = cudf.DataFrame({
            "country": [
                "France",
                "Spain",
                "Italy",
                "Germany",
                "United Kingdom",
                "China",
                "Japan",
                "India",
                "Brazil",
                "United States",
            ]
        })

    if shuffle:
        source_df = source_df.sample(frac=1, ignore_index=True)

    completion_task = {"task_type": "completion", "task_dict": {"input_keys": ["country"], }}

    pipe = LinearPipeline(config)

    pipe.set_source(InMemorySourceStage(config, dataframes=[source_df], repeat=repeat_count))

    pipe.add_stage(
        DeserializeStage(config, message_type=ControlMessage, task_type="llm_engine", task_payload=completion_task))

    pipe.add_stage(MonitorStage(config, description="Source rate", unit='questions'))

    pipe.add_stage(LLMEngineStage(config, engine=_build_engine(llm_service=llm_service)))

    pipe.add_stage(MonitorStage(config, description="Inference rate", unit="req", delayed_start=True))

    sink = pipe.add_stage(InMemorySinkStage(config))

    start_time = time.time()

    pipe.run()

    messages = sink.get_messages()
    responses = concat_dataframes(messages)
    logger.info("Pipeline complete. Received %s responses\n%s", len(messages), responses['response'])

    return start_time
