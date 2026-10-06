import logging

from sensai.core.bus import EventBus
from sensai.core.contracts import EventSink
from sensai.core.events import RUN_COMPLETED, RUN_STARTED, make_event
from sensai.core.fakes import CollectingEventSink


class BrokenSink:
    def emit(self, event):
        raise RuntimeError("sink down")


def test_bus_satisfies_the_event_sink_protocol():
    assert isinstance(EventBus(), EventSink)


def test_every_sink_receives_every_event_in_order():
    first, second = CollectingEventSink(), CollectingEventSink()
    bus = EventBus([first])
    bus.subscribe(second)
    events = [make_event(RUN_STARTED, "r1"), make_event(RUN_COMPLETED, "r1")]
    for event in events:
        bus.emit(event)
    assert first.events == events
    assert second.events == events


def test_a_failing_sink_does_not_stop_the_others(caplog):
    healthy = CollectingEventSink()
    bus = EventBus([BrokenSink(), healthy])
    event = make_event(RUN_STARTED, "r1")
    with caplog.at_level(logging.ERROR):
        bus.emit(event)
    assert healthy.events == [event]
    assert "run.started" in caplog.text
