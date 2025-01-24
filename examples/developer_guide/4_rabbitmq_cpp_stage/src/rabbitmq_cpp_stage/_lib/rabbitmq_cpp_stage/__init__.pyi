from __future__ import annotations
import datetime
import mrc.core.segment
__all__ = ['RabbitMQSourceStage']
class RabbitMQSourceStage(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, host: str, exchange: str, exchange_type: str = 'fanout', queue_name: str = '', poll_interval: datetime.timedelta = datetime.timedelta(microseconds=100000)) -> None:
        ...
