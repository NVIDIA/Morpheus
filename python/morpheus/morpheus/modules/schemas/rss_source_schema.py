# Copyright (c) 2022-2025, NVIDIA CORPORATION.
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
from typing import List

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

logger = logging.getLogger(__name__)


class RSSSourceSchema(BaseModel):
    feed_input: List[str] = Field(default_factory=list)
    run_indefinitely: bool = True
    batch_size: int = 128
    enable_cache: bool = False
    cache_dir: str = "./.cache/http"
    cooldown_interval_sec: int = 600
    request_timeout_sec: float = 2.0
    interval_sec: int = 600
    stop_after_rec: int = 0
    strip_markup: bool = True
    model_config = ConfigDict(extra='forbid')
