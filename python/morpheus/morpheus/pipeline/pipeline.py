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
import logging
import os
import signal
import sys
import threading
import typing
from collections import OrderedDict
from collections import defaultdict
from enum import Enum
from functools import partial

import mrc
import networkx
from tqdm import tqdm

import morpheus.pipeline as _pipeline  # pylint: disable=cyclic-import
from morpheus.config import Config
from morpheus.utils.type_utils import pretty_print_type_name

logger = logging.getLogger(__name__)

StageT = typing.TypeVar("StageT", bound=_pipeline.StageBase)


class PipelineState(Enum):
    INITIALIZED = "initialized"
    BUILT = "built"
    STARTED = "started"
    STOPPED = "stopped"
    COMPLETED = "completed"


class Pipeline():
    """
    Class for building your pipeline. A pipeline for your use case can be constructed by first adding a
    `Source` via `set_source` then any number of downstream `Stage` classes via `add_stage`. The order stages
    are added with `add_stage` determines the order in which stage executions are carried out. You can use
    stages included within Morpheus or your own custom-built stages.

    Parameters
    ----------
    config : `morpheus.config.Config`
        Pipeline configuration instance.

    """

    def __init__(self, config: Config):
        config.freeze()

        self._mutex = threading.RLock()

        self._source_count: int = None  # Maximum number of iterations for progress reporting. None = Unknown/Unlimited

        self._id_counter = 0
        self._num_threads = config.num_threads

        # Complete set of nodes across segments in this pipeline
        self._stages: typing.List[_pipeline.Stage] = []

        # Complete set of sources across segments in this pipeline
        self._sources: typing.List[_pipeline.SourceStage] = []

        # Dictionary containing segment information for this pipeline
        self._segments: typing.Dict = defaultdict(lambda: {"nodes": set(), "ingress_ports": [], "egress_ports": []})

        self.batch_size = config.pipeline_batch_size
        self.edge_buffer_size = config.edge_buffer_size

        self._segment_graphs = defaultdict(lambda: networkx.DiGraph())

        self._state = PipelineState.INITIALIZED

        self._mrc_executor: mrc.Executor = None

        self._loop: asyncio.AbstractEventLoop = None

        # Future that allows post_start to propagate exceptions back to pipeline
        self._post_start_future: asyncio.Future = None

        self._execution_mode = config.execution_mode

    @property
    def state(self) -> PipelineState:
        return self._state

    def _assert_not_built(self):
        assert self._state == PipelineState.INITIALIZED, "Pipeline has already been built. Cannot modify pipeline."

    def add_stage(self, stage: StageT, segment_id: str = "main") -> StageT:
        """
        Add a stage to a segment in the pipeline.

        Parameters
        ----------
        stage : Stage
            The stage object to add. It cannot be already added to another `Pipeline` object.

        segment_id : str
            ID indicating what segment the stage should be added to.
        """
        self._assert_not_built()
        assert stage._pipeline is None or stage._pipeline is self, "A stage can only be added to one pipeline at a time"

        segment_nodes = self._segments[segment_id]["nodes"]
        segment_graph = self._segment_graphs[segment_id]

        # Add to list of stages if it's a stage, not a source
        if (isinstance(stage, _pipeline.Stage)):
            segment_nodes.add(stage)
            self._stages.append(stage)
        elif (isinstance(stage, _pipeline.SourceStage)):
            segment_nodes.add(stage)
            self._sources.append(stage)
        else:
            raise NotImplementedError(f"add_stage() failed. Unknown node type: {type(stage)}")

        stage._pipeline = self

        segment_graph.add_node(stage)

        return stage

    def add_edge(self,
                 start: typing.Union[_pipeline.StageBase, _pipeline.Sender],
                 end: typing.Union[_pipeline.Stage, _pipeline.Receiver],
                 segment_id: str = "main"):
        """
        Create an edge between two stages and add it to a segment in the pipeline.
        When `start` and `end` are stages, they must have exactly one output and input port respectively.

        Parameters
        ----------
        start : typing.Union[StageBase, Sender]
            The start of the edge or parent stage.

        end : typing.Union[Stage, Receiver]
            The end of the edge or child stage.

        segment_id : str
            ID indicating what segment the edge should be added to.
        """
        self._assert_not_built()

        if (isinstance(start, _pipeline.StageBase)):
            assert len(start.output_ports) > 0, \
                "Cannot call `add_edge` with a stage with no output ports as the `start` parameter"
            assert len(start.output_ports) == 1, \
                ("Cannot call `add_edge` with a stage with with multiple output ports as the `start` parameter, "
                 "instead `add_edge` must be called for each output port individually.")
            start_port = start.output_ports[0]

        elif (isinstance(start, _pipeline.Sender)):
            start_port = start

        if (isinstance(end, _pipeline.Stage)):
            assert len(end.input_ports) > 0, \
                "Cannot call `add_edge` with a stage with no input ports as the `end` parameter"
            assert len(end.input_ports) == 1, \
                ("Cannot call `add_edge` with a stage with with multiple input ports as the `end` parameter, "
                 "instead `add_edge` must be called for each input port individually.")
            end_port = end.input_ports[0]

        elif (isinstance(end, _pipeline.Receiver)):
            end_port = end

        start_port._output_receivers.append(end_port)
        end_port._input_senders.append(start_port)

        segment_graph = self._segment_graphs[segment_id]
        segment_graph.add_edge(start_port.parent,
                               end_port.parent,
                               start_port_idx=start_port.port_number,
                               end_port_idx=end_port.port_number)

    def add_segment_edge(self,
                         egress_stage: _pipeline.BoundaryStageMixin,
                         egress_segment: str,
                         ingress_stage: _pipeline.BoundaryStageMixin,
                         ingress_segment: str,
                         port_pair: typing.Union[str, typing.Tuple[str, typing.Type, bool]]):
        """
        Create an edge between two segments in the pipeline.

        Parameters
        ----------

        egress_stage : Stage
            The egress stage of the parent segment

        egress_segment : str
            Segment ID of the parent segment

        ingress_stage : Stage
            The ingress stage of the child segment

        ingress_segment : str
            Segment ID of the child segment

        port_pair : typing.Union[str, typing.Tuple]
            Either the ID of the egress segment, or a tuple with the following three elements:
                * str: ID of the egress segment
                * class: type being sent (typically `object`)
                * bool: If the type is a shared pointer (typically should be `False`)
        """
        self._assert_not_built()
        assert isinstance(egress_stage, _pipeline.BoundaryStageMixin), "Egress stage must be a BoundaryStageMixin"
        egress_edges = self._segments[egress_segment]["egress_ports"]
        egress_edges.append({
            "port_pair": port_pair,
            "input_sender": egress_stage.unique_name,
            "output_receiver": ingress_stage.unique_name,
            "receiver_segment": ingress_segment
        })

        assert isinstance(ingress_stage, _pipeline.BoundaryStageMixin), "Ingress stage must be a BoundaryStageMixin"
        ingress_edges = self._segments[ingress_segment]["ingress_ports"]
        ingress_edges.append({
            "port_pair": port_pair,
            "input_sender": egress_stage.unique_name,
            "sender_segment": egress_segment,
            "output_receiver": ingress_stage.unique_name
        })

    def _pre_build(self):
        assert len(self._sources) > 0, "Pipeline must have a source stage"

        logger.info("====Pipeline Pre-build====")

        for segment_id in self._segments.keys():
            logger.info("====Pre-Building Segment: %s====", segment_id)
            segment_graph = self._segment_graphs[segment_id]

            # Check if preallocated columns are requested, this needs to happen before the source stages are built
            needed_columns = OrderedDict()
            preallocator_stages = []

            # This should be a BFS search from each source nodes; but, since we don't have source stage loops
            # topo_sort provides a reasonable approximation.
            for stage in networkx.topological_sort(segment_graph):
                needed_columns.update(stage.get_needed_columns())
                if (isinstance(stage, _pipeline.PreallocatorMixin)):
                    preallocator_stages.append(stage)

                if (stage.can_pre_build()):
                    stage._pre_build()

            if (len(needed_columns) > 0):
                for stage in preallocator_stages:
                    stage.set_needed_columns(needed_columns)

            if (not all(x.is_pre_built for x in segment_graph.nodes())):
                logger.warning("Cyclic pipeline graph detected! Building with reduced constraints")

                for stage in segment_graph.nodes():
                    if (stage.can_pre_build(check_ports=True)):
                        stage._pre_build()
                    else:
                        raise RuntimeError("Could not build pipeline. Ensure all types can be determined")

            # Finally, execute the link phase (only necessary for circular pipelines)
            # for s in source_and_stages:
            for stage in segment_graph.nodes():
                for port in typing.cast(_pipeline.StageBase, stage).input_ports:
                    port.link_schema()

            logger.info("====Pre-Building Segment Complete!====")

        logger.info("====Pipeline Pre-build Complete!====")

    def build(self):
        """
        This function sequentially activates all the Morpheus pipeline stages passed by the users to execute a
        pipeline. For the `Source` and all added `Stage` objects, `StageBase.build` will be called sequentially to
        construct the pipeline.

        Once the pipeline has been constructed, this will start the pipeline by calling `Source.start` on the source
        object.
        """
        assert self._state == PipelineState.INITIALIZED, "Pipeline can only be built once!"
        assert len(self._sources) > 0, "Pipeline must have a source stage"

        self._pre_build()

        logger.info("====Registering Pipeline====")

        # Set the default channel size
        mrc.Config.default_channel_size = self.edge_buffer_size

        exec_options = mrc.Options()
        exec_options.topology.user_cpuset = f"0-{self._num_threads - 1}"
        exec_options.engine_factories.default_engine_type = mrc.core.options.EngineType.Thread

        self._mrc_executor = mrc.Executor(exec_options)

        mrc_pipeline = mrc.Pipeline()

        def inner_build(builder: mrc.Builder, segment_id: str):
            logger.info("====Building Segment: %s====", segment_id)
            segment_graph = self._segment_graphs[segment_id]

            # This should be a BFS search from each source nodes; but, since we don't have source stage loops
            # topo_sort provides a reasonable approximation.
            for stage in networkx.topological_sort(segment_graph):
                if (stage.can_build()):
                    stage.build(builder)

            if (not all(x.is_built for x in segment_graph.nodes())):
                logger.warning("Cyclic pipeline graph detected! Building with reduced constraints")

                for stage in segment_graph.nodes():
                    if (stage.can_build(check_ports=True)):
                        stage.build(builder)

            if (not all(x.is_built for x in segment_graph.nodes())):
                raise RuntimeError("Could not build pipeline. Ensure all types can be determined")

            # Finally, execute the link phase (only necessary for circular pipelines)
            for stage in segment_graph.nodes():
                for port in typing.cast(_pipeline.StageBase, stage).input_ports:
                    port.link_node(builder=builder)

            # Call the start method for the stages in this segment. Must run on the loop and wait for the result
            asyncio.run_coroutine_threadsafe(self._async_start(segment_graph.nodes()), self._loop).result()

            logger.info("====Building Segment Complete!====")

        logger.info("====Building Pipeline====")
        for segment_id, segment in self._segments.items():
            segment_ingress_ports = segment["ingress_ports"]
            segment_egress_ports = segment["egress_ports"]
            segment_inner_build = partial(inner_build, segment_id=segment_id)

            mrc_pipeline.make_segment(segment_id, [port_info["port_pair"] for port_info in segment_ingress_ports],
                                      [port_info["port_pair"] for port_info in segment_egress_ports],
                                      segment_inner_build)

        logger.info("====Building Pipeline Complete!====")

        self._mrc_executor.register_pipeline(mrc_pipeline)

        with self._mutex:
            self._state = PipelineState.BUILT

        logger.info("====Registering Pipeline Complete!====")

    async def _start(self):
        assert self._state == PipelineState.BUILT, "Pipeline must be built before starting"

        with self._mutex:
            self._state = PipelineState.STARTED

        # Save off the current loop so we can use it in async_start
        self._loop = asyncio.get_running_loop()

        # Setup error handling and cancellation of the pipeline
        def error_handler(_, context: dict):

            msg = f"Unhandled exception in async loop! Exception: \n{context['message']}"
            exception = context.get("exception", Exception())

            logger.critical(msg, exc_info=exception)

        self._loop.set_exception_handler(error_handler)

        exit_count = 0

        # Handles Ctrl+C for graceful shutdown
        def term_signal():

            nonlocal exit_count
            exit_count = exit_count + 1

            if (exit_count == 1):
                tqdm.write("Stopping pipeline. Please wait... Press Ctrl+C again to kill.")
                self.stop()
            else:
                tqdm.write("Killing")
                sys.exit(1)

        for sig in [signal.SIGINT, signal.SIGTERM]:
            self._loop.add_signal_handler(sig, term_signal)

        logger.info("====Starting Pipeline====")

        self._mrc_executor.start()

        logger.info("====Pipeline Started====")

        async def post_start(executor):

            try:
                # Make a local reference so the object doesn't go out of scope from a call to stop()
                await executor.join_async()
            except Exception:
                logger.exception("Exception occurred in pipeline. Rethrowing")
                raise
            finally:
                # Call join on all sources. This only occurs after all messages have been processed fully.
                for source in list(self._sources):
                    await source.join()

                # Now call join on all stages
                for stage in list(self._stages):
                    await stage.join()

                self._on_stop()

                with self._mutex:
                    self._state = PipelineState.COMPLETED

        self._post_start_future = asyncio.create_task(post_start(self._mrc_executor))

    def stop(self):
        """
        Stops all running stages and the underlying MRC pipeline.
        """
        assert self._state == PipelineState.STARTED, "Pipeline must be running to stop it"

        logger.info("====Stopping Pipeline====")
        for stage in list(self._sources) + list(self._stages):
            stage.stop()

        self._mrc_executor.stop()

        with self._mutex:
            self._state = PipelineState.STOPPED

        logger.info("====Pipeline Stopped====")
        self._on_stop()

    async def join(self):
        """
        Wait until pipeline completes upon which join methods of sources and stages will be called.
        """
        assert self._post_start_future is not None, "Pipeline must be started before joining"

        await self._post_start_future

    def _on_stop(self):
        self._mrc_executor = None

    async def build_and_start(self):

        if (self._state == PipelineState.INITIALIZED):
            try:
                self.build()
            except Exception:
                logger.exception("Error occurred during Pipeline.build(). Exiting.", exc_info=True)
                raise

        await self._start()

    async def _async_start(self, stages: networkx.classes.reportviews.NodeView):
        # This method is called once for each segment in the pipeline executed on this host
        for stage in stages:
            await stage.start_async()

    def visualize(self, filename: str = None, **graph_kwargs):
        """
        Output a pipeline diagram to `filename`. The file format of the diagrame is inferred by the extension of
        `filename`. If the directory path leading to `filename` does not exist it will be created, if `filename` already
        exists it will be overwritten.  Requires the graphviz library.
        """

        if self._state == PipelineState.INITIALIZED:
            raise RuntimeError("Pipeline.visualize() requires that the Pipeline has been started before generating "
                               "the visualization. Please call Pipeline.build() or  Pipeline.run() before calling "
                               "Pipeline.visualize().")

        # Mimic the streamz visualization
        # 1. Create graph (already done since we use networkx under the hood)
        # 2. Readable graph
        # 3. To graphviz
        # 4. Draw
        import graphviz

        # Default graph attributes
        graph_attr = {
            "nodesep": "1",
            "ranksep": "1",
            "pad": "0.5",
        }

        # Allow user to overwrite defaults
        graph_attr.update(graph_kwargs)

        gv_graph = graphviz.Digraph(graph_attr=graph_attr)
        gv_graph.attr(compound="true")
        gv_subgraphs = {}

        # Need a little different functionality for left/right vs vertical
        is_lr = graph_kwargs.get("rankdir", None) == "LR"

        start_def_port = ":e" if is_lr else ":s"
        end_def_port = ":w" if is_lr else ":n"

        def has_ports(node: _pipeline.StageBase, is_input):
            if (is_input):
                return len(node.input_ports) > 0

            return len(node.output_ports) > 0

        # Now build up the nodes
        for segment_id in self._segments:
            gv_subgraphs[segment_id] = graphviz.Digraph(f"cluster_{segment_id}")
            gv_subgraph = gv_subgraphs[segment_id]
            gv_subgraph.attr(label=segment_id)
            for name, attrs in typing.cast(typing.Mapping[_pipeline.StageBase, dict],
                                           self._segment_graphs[segment_id].nodes).items():
                node_attrs = attrs.copy()

                label = ""

                show_in_ports = has_ports(name, is_input=True)
                show_out_ports = has_ports(name, is_input=False)

                # Build the ports for the node. Only show ports if there are any
                # (Would like to have this not show for one port, but the lines get all messed up)
                if (show_in_ports):
                    tmp_str = " | ".join([f"<u{x.port_number}> input_port: {x.port_number}" for x in name.input_ports])
                    in_port_label = f" {{ {tmp_str} }} | "
                    label += in_port_label

                label += name.unique_name

                if (show_out_ports):
                    tmp_str = " | ".join(
                        [f"<d{x.port_number}> output_port: {x.port_number}" for x in name.output_ports])
                    out_port_label = f" | {{ {tmp_str} }}"
                    label += out_port_label

                if (show_in_ports or show_out_ports):
                    label = f"{{ {label} }}"

                node_attrs.update({
                    "label": label,
                    "shape": "record",
                    "fillcolor": "white",
                })
                # TODO(MDD): Eventually allow nodes to have different attributes based on type
                # node_attrs.update(n.get_graphviz_attrs())
                gv_subgraph.node(name.unique_name, **node_attrs)

        # Build up edges
        for segment_id in self._segments:
            gv_subgraph = gv_subgraphs[segment_id]
            for e, attrs in typing.cast(typing.Mapping[typing.Tuple[_pipeline.StageBase, _pipeline.StageBase], dict],
                                        self._segment_graphs[segment_id].edges()).items():  # noqa: E501

                edge_attrs = {}

                start_name = e[0].unique_name

                # Append the port if necessary
                if (has_ports(e[0], is_input=False)):
                    start_name += f":d{attrs['start_port_idx']}"
                else:
                    start_name += start_def_port

                end_name = e[1].unique_name

                if (has_ports(e[1], is_input=True)):
                    end_name += f":u{attrs['end_port_idx']}"
                else:
                    end_name += end_def_port

                # Now we only want to show the type label in some scenarios:
                # 1. If there is only one edge between two nodes, draw type in middle "label"
                # 2. For port with an edge, only draw that port's type once (using index 0 of the senders/receivers)
                start_port_idx = int(attrs['start_port_idx'])
                end_port_idx = int(attrs['end_port_idx'])

                out_port = e[0].output_ports[start_port_idx]
                in_port = e[1].input_ports[end_port_idx]

                # Check for situation #1
                if (len(in_port._input_senders) == 1 and len(out_port._output_receivers) == 1
                        and (in_port.input_schema == out_port.output_schema)):

                    edge_attrs["label"] = pretty_print_type_name(in_port.input_type)
                else:
                    rec_idx = out_port._output_receivers.index(in_port)
                    sen_idx = in_port._input_senders.index(out_port)

                    # Add type labels if available
                    if (rec_idx == 0 and out_port.output_schema is not None):
                        edge_attrs["taillabel"] = pretty_print_type_name(out_port.output_type)

                    if (sen_idx == 0 and in_port.input_schema is not None):
                        edge_attrs["headlabel"] = pretty_print_type_name(in_port.input_type)

                gv_subgraph.edge(start_name, end_name, **edge_attrs)

            for egress_port in self._segments[segment_id]["egress_ports"]:
                gv_graph.edge(egress_port["input_sender"],
                              egress_port["output_receiver"],
                              style="dashed",
                              label=f"Segment Port: {egress_port['port_pair'][0]}")

        for gv_subgraph in gv_subgraphs.values():
            gv_graph.subgraph(gv_subgraph)

        file_format = os.path.splitext(filename)[-1].replace(".", "")

        viz_binary = gv_graph.pipe(format=file_format)
        # print(gv_graph.source)

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, "wb") as f:
            f.write(viz_binary)

    async def run_async(self):
        """
        This function sets up the current asyncio loop, builds the pipeline, and awaits on it to complete.
        """
        try:
            await self.build_and_start()
            await self.join()

        except KeyboardInterrupt:
            tqdm.write("Stopping pipeline. Please wait...")

            # Stop the pipeline
            self.stop()

        finally:
            # Shutdown the async generator sources and exit
            logger.info("====Pipeline Complete====")

    def run(self):
        """
        This function makes use of asyncio features to keep the pipeline running indefinitely.
        """

        # Use asyncio.run() to launch the pipeline. This creates and destroys an event loop so re-running a pipeline in
        # the same process wont fail
        asyncio.run(self.run_async())
