# Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

import asyncio
import json
import logging
import os
import sys
import typing

import mrc
import mrc.core.operators as ops
import numpy as np
import pandas as pd
import pyarrow as pa
import websockets.legacy.server
from websockets.server import serve  # pylint: disable=no-name-in-module

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.messages import ControlMessage
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.pass_thru_type_mixin import PassThruTypeMixin
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.utils.producer_consumer_queue import AsyncIOProducerConsumerQueue
from morpheus.utils.producer_consumer_queue import Closed
from morpheus.utils.type_aliases import DataFrameType

logger = logging.getLogger(__name__)


@register_stage("gen-viz", modes=[PipelineModes.NLP], command_args={"deprecated": True})
class GenerateVizFramesStage(GpuAndCpuMixin, PassThruTypeMixin, SinglePortStage):
    """
    Write out visualization DataFrames.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance
    out_dir : str
        Output directory to write visualization frames
    overwrite : bool
        Overwrite file if exists

    """

    def __init__(self,
                 c: Config,
                 server_url: str = "0.0.0.0",
                 server_port: int = 8765,
                 out_dir: str = None,
                 overwrite: bool = False):
        super().__init__(c)

        self._server_url = server_url
        self._server_port = server_port
        self._out_dir = out_dir
        self._overwrite = overwrite

        self._first_timestamp = -1
        self._buffers = []
        self._buffer_queue: AsyncIOProducerConsumerQueue = None

        self._replay_buffer = []

        # Properties set on start
        self._loop: asyncio.AbstractEventLoop = None
        self._server_task: asyncio.Task = None
        self._server_close_event: asyncio.Event = None

        self._df_class: type[DataFrameType] = self.get_df_class()

    @property
    def name(self) -> str:
        return "gen_viz"

    def accepted_types(self) -> typing.Tuple:
        """
        Returns accepted input types for this stage.

        Returns
        -------
        typing.Tuple[ControlMessage]
            Accepted input types

        """
        return (ControlMessage, )

    def supports_cpp_node(self):
        return False

    @staticmethod
    def round_to_sec(x: int | float):
        """
        Round to even seconds second

        Parameters
        ----------
        x : int | float
            Rounding up the value

        Returns
        -------
        int
            Value rounded up

        """
        return int(round(x / 1000.0) * 1000)

    def _to_vis_df(self, msg: ControlMessage):

        idx2label = {
            0: 'address',
            1: 'bank_acct',
            2: 'credit_card',
            3: 'email',
            4: 'govt_id',
            5: 'name',
            6: 'password',
            7: 'phone_num',
            8: 'secret_keys',
            9: 'user'
        }

        columns = ["timestamp", "src_ip", "dest_ip", "src_port", "dest_port", "data"]
        df = msg.payload().get_data(columns)

        def indent_data(y: str):
            try:
                return json.dumps(json.loads(y), indent=3)
            except Exception:
                return y

        if not isinstance(df, pd.DataFrame):
            df = df.to_pandas()

        df["data"] = df["data"].apply(indent_data)

        probs = msg.tensors().get_tensor("probs")

        pass_thresh = (probs >= 0.5).any(axis=1)
        max_arg = probs.argmax(axis=1)

        condlist = [pass_thresh]

        choicelist = [max_arg]

        index_sens_info = np.select(condlist, choicelist, default=len(idx2label))

        df["si"] = pd.Series(np.choose(index_sens_info.get(), list(idx2label.values()) + ["none"]).tolist())

        df["ts_round_sec"] = (df["timestamp"] / 1000.0).astype(int) * 1000

        # Return a list of tuples of (ts_round_sec, dataframe)
        return list(df.groupby(df.ts_round_sec))

    def _write_viz_file(self, x: typing.List[typing.Tuple[int, pd.DataFrame]]):

        curr_timestamp = x[0][0]

        in_df = pd.concat([df for _, df in x], ignore_index=True).sort_values(by=["timestamp"])

        if (self._first_timestamp == -1):
            self._first_timestamp = curr_timestamp

        offset = (curr_timestamp - self._first_timestamp) / 1000

        out_file = os.path.join(self._out_dir, f"{offset}.csv")

        assert self._overwrite or not os.path.exists(out_file)

        in_df.to_csv(out_file, columns=["timestamp", "src_ip", "dest_ip", "src_port", "dest_port", "si", "data"])

    async def start_async(self):
        """
        Launch the Websocket server and asynchronously send messages via Websocket.
        """

        self._loop = asyncio.get_event_loop()

        self._buffer_queue = AsyncIOProducerConsumerQueue(maxsize=2)

        async def client_connected(websocket: websockets.legacy.server.WebSocketServerProtocol):
            """
            Establishes a connection with the WebSocket server.
            """

            logger.info("Got connection from: %s:%s", *websocket.remote_address)

            while True:
                try:
                    next_buffer = await self._buffer_queue.get()
                    await websocket.send(next_buffer.to_pybytes())
                except Closed:
                    break
                except Exception as ex:
                    if not self._server_close_event.is_set():
                        # Don't log if we are shutting down
                        logger.exception("Error occurred trying to send message over socket", exc_info=ex)

            logger.info("Disconnected from: %s:%s", *websocket.remote_address)

        async def run_server():
            """
            Runs Websocket server.
            """

            try:

                async with serve(client_connected, self._server_url, self._server_port) as server:

                    listening_on = [":".join([str(y) for y in x.getsockname()]) for x in server.sockets]
                    listening_on_str = [f"'{x}'" for x in listening_on]

                    logger.info("Websocket server listening at: %s", ", ".join(listening_on_str))

                    await self._server_close_event.wait()

                    logger.info("Server shut down")

                logger.info("Server shut down. Is queue empty: %s", self._buffer_queue.empty())
            except Exception as e:
                logger.error("Error during serve", exc_info=e)
                raise

        self._server_task = self._loop.create_task(run_server())

        self._server_close_event = asyncio.Event()

        await asyncio.sleep(1.0)

        return await super().start_async()

    async def join(self):
        """
        Stages can implement this to perform cleanup steps when pipeline is stopped.
        """

        if (self._loop is not None):
            asyncio.run_coroutine_threadsafe(self._stop_server(), loop=self._loop)
            pass

    async def _stop_server(self):

        # Only run this once
        if (self._buffer_queue.is_closed()):
            return

        logger.info("Shutting down queue")

        await self._buffer_queue.close()

        self._server_close_event.set()

        # Wait for it to fully shut down
        await self._server_task

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:

        def node_fn(input_obs, output_obs):

            def write_batch(msg: ControlMessage):

                sink = pa.BufferOutputStream()

                # This is the timestamp of the earliest message
                time0 = msg.payload().get_data("timestamp").min()

                columns = ["timestamp", "src_ip", "dest_ip", "secret_keys", "data"]
                df = msg.payload().get_data(columns)

                out_df = self._df_class()

                out_df["dt"] = (df["timestamp"] - time0).astype(np.int32)
                out_df["src"] = df["src_ip"].str.ip_to_int().astype(np.uint32)
                out_df["dst"] = df["dest_ip"].str.ip_to_int().astype(np.uint32)
                out_df["lvl"] = df["secret_keys"].astype(np.int32)
                out_df["data"] = df["data"]

                array_table = out_df.to_arrow()

                with pa.ipc.new_stream(sink, array_table.schema) as writer:
                    writer.write(array_table)

                out_buf = sink.getvalue()

                try:
                    # Enqueue the buffer and block until that completes
                    asyncio.run_coroutine_threadsafe(self._buffer_queue.put(out_buf), loop=self._loop).result()
                except Closed:
                    # Ignore closed errors. Likely the pipeline is shutting down
                    pass

            input_obs.pipe(ops.map(write_batch)).subscribe(output_obs)

            logger.info("Gen-viz stage completed. Waiting for shutdown")

            shutdown_future = asyncio.run_coroutine_threadsafe(self._stop_server(), loop=self._loop)

            # Wait for shutdown. Unless we have a debugger attached
            shutdown_future.result(timeout=2.0 if sys.gettrace() is None else None)

            logger.info("Gen-viz shutdown complete")

        # Sink to file
        to_filenode = builder.make_node(self.unique_name, ops.build(node_fn))
        builder.make_edge(input_node, to_filenode)

        return to_filenode
