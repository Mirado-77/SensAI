"""Event bus: delivers each event to every subscribed sink."""

import logging
from collections.abc import Iterable

from sensai.core.contracts import EventSink
from sensai.core.events import Event

logger = logging.getLogger(__name__)


class EventBus:
    """A failing sink is logged and skipped: it never breaks the other sinks or the run."""

    def __init__(self, sinks: Iterable[EventSink] = ()) -> None:
        self._sinks = list(sinks)

    def subscribe(self, sink: EventSink) -> None:
        self._sinks.append(sink)

    def emit(self, event: Event) -> None:
        for sink in self._sinks:
            try:
                sink.emit(event)
            except Exception:
                logger.exception("Event sink %r failed on %s", sink, event.type)
