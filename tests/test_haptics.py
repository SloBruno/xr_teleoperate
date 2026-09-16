import numpy as np

from teleop.utils.haptics import (
    HapticTransportAdapter,
    PressureHapticMapper,
    extract_dex3_pressure,
    pressure_to_haptic,
)


def test_invalid_pressure_and_maximum_fail_closed():
    assert pressure_to_haptic(np.array([np.nan, 1.0]), 10.0) == 0.0
    assert pressure_to_haptic(np.array([np.inf]), 10.0) == 0.0
    assert pressure_to_haptic(np.array([1.0]), 0.0) == 0.0


def test_zero_and_calibrated_maximum_map_to_zero_and_one():
    assert pressure_to_haptic(np.zeros(3), 10.0) == 0.0
    assert pressure_to_haptic(np.full(3, 10.0), 10.0) == 1.0


def test_finite_pressure_is_monotonic_and_bounded_with_deadband():
    values = [pressure_to_haptic(np.array([p]), 10.0) for p in range(0, 16)]
    assert values == sorted(values)
    assert all(np.isfinite(values))
    assert all(0.0 <= value <= 1.0 for value in values)
    assert pressure_to_haptic(np.array([0.5]), 10.0, deadband=1.0) == 0.0


def test_mapper_rate_limits_rising_and_falling_output():
    mapper = PressureHapticMapper(maximum=10.0, max_rate=0.5, max_age=0.25, clock=lambda: 0.0)
    assert mapper.update(np.array([10.0])) == 0.0
    mapper.clock = lambda: 1.0
    assert mapper.update(np.array([10.0])) == 0.5
    mapper.clock = lambda: 2.0
    assert mapper.update(np.array([0.0])) == 0.0
    assert mapper.update(np.array([10.0]), sample_age=0.26) == 0.0


def test_pressure_extraction_is_side_local_and_invalid_samples_are_zero():
    class Sensor:
        def __init__(self, values):
            self.pressure = values

    class HandState:
        def __init__(self, values):
            self.press_sensor_state = [Sensor(values)]

    assert extract_dex3_pressure(HandState([2.0] * 12)) == 2.0
    assert extract_dex3_pressure(HandState([7.0] * 12)) == 7.0
    assert extract_dex3_pressure(HandState([np.nan] * 12)) == 0.0


def test_pressure_sample_handoff_maps_only_fresh_finite_contact():
    now = [10.0]
    mapper = PressureHapticMapper(maximum=10.0, deadband=1.0, max_rate=100.0, max_age=0.25, clock=lambda: now[0])
    transport = HapticTransportAdapter(None, mapper_by_side={"left": mapper, "right": mapper}, clock=lambda: now[0])

    now[0] = 10.1
    assert transport.map_pressure("left", np.array([5.0]), sample_timestamp=10.0) == 4.0 / 9.0
    assert transport.map_pressure("left", np.array([np.nan]), sample_timestamp=9.9) == 0.0
    assert transport.map_pressure("left", np.array([5.0]), sample_timestamp=9.0) == 0.0


def test_motion_controller_haptic_upsert_has_exact_left_payload_and_unique_hash(monkeypatch):
    elements = []

    class FakeMotionControllers:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeUpsert:
        def __matmul__(self, element):
            elements.append(element)

    class FakeSession:
        upsert = FakeUpsert()

    import teleop.utils.haptics as haptics
    monkeypatch.setattr(haptics, "MotionControllers", FakeMotionControllers)
    transport = HapticTransportAdapter(FakeSession(), clock=lambda: 10.0, min_interval=0.0)

    assert transport.emit("left", 0.5, duration_ms=80) is True
    assert transport.emit("left", 0.75, duration_ms=80) is True
    assert [element.kwargs for element in elements] == [
        {
            "key": "motionControllers", "left": True, "right": True,
            "pulseLeftStrength": 0.5, "pulseLeftDuration": 80,
            "puseLeftHash": elements[0].kwargs["puseLeftHash"],
        },
        {
            "key": "motionControllers", "left": True, "right": True,
            "pulseLeftStrength": 0.75, "pulseLeftDuration": 80,
            "puseLeftHash": elements[1].kwargs["puseLeftHash"],
        },
    ]
    assert elements[0].kwargs["puseLeftHash"] != elements[1].kwargs["puseLeftHash"]

    elements.clear()
    assert transport.emit("right", 0.25, duration_ms=90) is True
    assert elements[0].kwargs == {
        "key": "motionControllers", "left": True, "right": True,
        "pulseRightStrength": 0.25, "pulseRightDuration": 90,
        "puseRightHash": elements[0].kwargs["puseRightHash"],
    }


def test_haptic_upsert_is_suppressed_for_invalid_stale_zero_or_missing_session(monkeypatch):
    calls = []

    class FakeMotionControllers:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    class FakeUpsert:
        def __matmul__(self, element):
            calls.append(element)

    class FakeSession:
        upsert = FakeUpsert()

    import teleop.utils.haptics as haptics
    monkeypatch.setattr(haptics, "MotionControllers", FakeMotionControllers)
    transport = HapticTransportAdapter(FakeSession(), clock=lambda: 10.0, min_interval=0.0)
    assert transport.emit("left", 0.0, duration_ms=80) is False
    assert transport.emit("left", np.nan, duration_ms=80) is False
    assert transport.emit("unknown", 0.5, duration_ms=80) is False
    assert transport.map_pressure("right", np.array([5.0]), sample_timestamp=9.0) == 0.0
    assert HapticTransportAdapter(None).emit("left", 0.5, duration_ms=80) is False
    assert calls == []
    assert transport.limit_duration(5000) == 250


def test_timestamped_zero_after_contact_suppresses_haptic_pulse(monkeypatch):
    elements = []

    class FakeMotionControllers:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeUpsert:
        def __matmul__(self, element):
            elements.append(element)

    class FakeSession:
        upsert = FakeUpsert()

    import teleop.utils.haptics as haptics
    monkeypatch.setattr(haptics, "MotionControllers", FakeMotionControllers)
    now = [10.0]
    transport = HapticTransportAdapter(
        FakeSession(), max_rate=100.0, min_interval=0.0, clock=lambda: now[0]
    )

    now[0] = 10.1
    assert transport.emit_pressure("left", np.array([10.0]), 10.1, 80) is True
    now[0] = 10.11
    assert transport.emit_pressure("left", np.array([0.0]), 10.11, 80) is False
    assert len(elements) == 1
