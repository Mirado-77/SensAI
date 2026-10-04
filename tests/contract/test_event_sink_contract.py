from sensai.core.contracts import EventSink
from sensai.core.events import MODEL_CHUNK, RUN_COMPLETED, RUN_STARTED, make_event


def test_satisfies_the_protocol(sink_harness):
    assert isinstance(sink_harness.sink, EventSink)


def test_emit_is_synchronous(sink_harness):
    assert sink_harness.sink.emit(make_event(RUN_STARTED, "r1")) is None


def test_keeps_events_in_order(sink_harness):
    events = [
        make_event(RUN_STARTED, "r1"),
        make_event(MODEL_CHUNK, "r1", {"text": "a"}),
        make_event(RUN_COMPLETED, "r1"),
    ]
    for event in events:
        sink_harness.sink.emit(event)
    assert sink_harness.received() == events
