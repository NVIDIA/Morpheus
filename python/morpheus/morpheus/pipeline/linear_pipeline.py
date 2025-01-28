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

import logging
import typing

import morpheus.pipeline as _pipeline  # pylint: disable=cyclic-import
from morpheus.config import Config

SinglePortStageT = typing.TypeVar("SinglePortStageT", bound=_pipeline.SinglePortStage)
SourceT = typing.TypeVar("SourceT", bound=_pipeline.SourceStage)

logger = logging.getLogger(__name__)


class LinearPipeline(_pipeline.Pipeline):
    """
    This class is used to build linear pipelines where we have a single output source stage followed by stages that are
    executed sequentially in the order they were added.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    """

    def __init__(self, c: Config):
        super().__init__(c)

        self._current_segment_id = ""
        self._next_segment_index = 0
        self._increment_segment_id()

        self._linear_stages: typing.List[_pipeline.StageBase] = []

    def _increment_segment_id(self):
        self._linear_stages = []
        self._current_segment_id = f"linear_segment_{self._next_segment_index}"
        self._next_segment_index += 1

    def set_source(self, source: SourceT) -> SourceT:
        """
        Set a pipeline's source stage to consume messages before it begins executing stages. This must be
        called once before calling `run` or `run_async`.

        Parameters
        ----------
        source : `SourceStage`
            The source stage wraps the implementation in a stream that allows it to read from Kafka or a file.

        """

        if (len(self._sources) > 0 and source not in self._sources):
            logger.warning(
                "LinearPipeline already has a source. Setting a new source will clear out all existing stages")

            self._sources.clear()

        if (len(self._linear_stages) > 0):
            # TODO(devin): This doesn't seem right, if we add another source, our underlying Pipeline could still have
            # any number of dangling nodes.
            logger.warning("Clearing %d stages from pipeline", len(self._linear_stages))
            self._linear_stages.clear()

        # Need to store the source in the pipeline
        super().add_stage(source, self._current_segment_id)

        # Store this as the first one in the linear stages. Must be index 0
        self._linear_stages.append(source)

        return source

    def add_stage(self, stage: SinglePortStageT) -> SinglePortStageT:  # pylint:disable=arguments-differ
        """
        Add a stage to the pipeline. All `Stage` classes added with this method will be executed sequentially
        inthe order they were added.

        Parameters
        ----------
        stage : `Stage`
            The stage object to add. It cannot be already added to another `Pipeline` object.

        """

        assert len(self._linear_stages) > 0, "A source must be set on a LinearPipeline before adding any stages"
        assert isinstance(stage, _pipeline.SinglePortStage), ("Only `SinglePortStage` stages are accepted in "
                                                              "`add_stage()`")

        # Add this stage to the segment graph
        super().add_stage(stage, self._current_segment_id)

        # Make an edge between the last node and this one
        super().add_edge(self._linear_stages[-1], stage, self._current_segment_id)

        self._linear_stages.append(stage)

        return stage

    def add_segment_boundary(self, data_type=None, as_shared_pointer=False):
        """
        Parameters
        ----------
        data_type : `data_type`
            data type that will be passed across the segment boundary, defaults to a general python object
            if 'data_type' has no registered edge adapters.

        as_shared_pointer : `boolean`
            Whether the data type will be wrapped in a shared pointer. Currently this is not implemented.

        Examples
        --------
        >>> # Create a config and pipeline
        >>> config = Config()
        >>> pipe = LinearPipeline(config)
        >>>
        >>> # Add a source in Segment #1
        >>> pipe.set_source(FileSourceStage(config, filename=val_file_name, iterative=False))
        >>>
        >>> # Add a segment boundary
        >>> # [Current Segment] - [Egress Boundary] ---- [Ingress Boundary] - [Next Segment]
        >>> pipe.add_segment_boundary(MessageMeta)
        >>>
        >>> # Add a sink in Segment #2
        >>> pipe.add_stage(WriteToFileStage(config, filename=out_file, overwrite=False))
        >>>
        >>> pipe.run()
        """

        # Local imports to avoid circular dependencies
        # pylint:disable=cyclic-import
        from morpheus.stages.boundary.linear_boundary_stage import LinearBoundaryEgressStage
        from morpheus.stages.boundary.linear_boundary_stage import LinearBoundaryIngressStage

        assert as_shared_pointer is False, "Shared pointers are not currently supported"

        if (len(self._linear_stages) == 0):
            raise RuntimeError("Cannot create a segment boundary, current segment is empty.")

        empty_config = Config()
        empty_config.execution_mode = self._execution_mode
        boundary_egress = LinearBoundaryEgressStage(empty_config,
                                                    boundary_port_id=self._current_segment_id,
                                                    data_type=data_type)
        boundary_ingress = LinearBoundaryIngressStage(empty_config,
                                                      boundary_port_id=self._current_segment_id,
                                                      data_type=data_type)

        # TODO: update to use data_type once typeid is attached to registered objects out of band # pylint:disable=fixme
        # https://github.com/nv-morpheus/MRC/issues/176
        port_id_tuple = (self._current_segment_id, object, False) if data_type else self._current_segment_id

        self.add_stage(boundary_egress)
        egress_segment_id = self._current_segment_id

        self._increment_segment_id()
        ingress_segment_id = self._current_segment_id

        self._linear_stages.append(boundary_ingress)

        super().add_stage(boundary_ingress, self._current_segment_id)
        super().add_segment_edge(boundary_egress,
                                 egress_segment_id,
                                 boundary_ingress,
                                 ingress_segment_id,
                                 port_id_tuple)

        return (boundary_egress, boundary_ingress)
