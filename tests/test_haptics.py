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


def test_unsupported_session_transport_is_an_explicit_noop():
    session = object()
    transport = HapticTransportAdapter(session)
    assert transport.supported is False
    assert transport.emit("left", 1.0, duration_ms=100) is False
    assert transport.limit_duration(5000) == 250
