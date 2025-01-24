from __future__ import annotations
import mrc.core.segment
__all__ = ['PassThruStage']
class PassThruStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str) -> None:
        ...
