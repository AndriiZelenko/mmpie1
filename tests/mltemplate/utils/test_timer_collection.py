"""Unit test methods for mmpie1.utils.timer_collection utility module."""
import time

import pytest

from mmpie1.utils import TimerCollection


def test_timer_collection():
    """Test the TimerCollection class."""
    timer_collection = TimerCollection()

    # Test that a timer can be started
    timer_collection.start("Timer 1")
    time.sleep(0.1)
    assert timer_collection.duration("Timer 1") > 0.0

    # Test that a timer can be stopped
    time.sleep(0.1)
    timer_collection.stop("Timer 1")
    assert timer_collection.duration("Timer 1") > 0.1

    # Test that a timer can be reset
    timer_collection.reset("Timer 1")
    assert timer_collection.duration("Timer 1") == 0.0

    # Test that a timer can be started and stopped multiple times
    timer_collection.start("Timer 1")
    time.sleep(0.1)
    assert timer_collection.duration("Timer 1") > 0.1
    timer_collection.stop("Timer 1")
    time.sleep(1.0)
    assert timer_collection.duration("Timer 1") < 1.0
    timer_collection.start("Timer 1")
    time.sleep(0.1)
    timer_collection.stop("Timer 1")
    assert timer_collection.duration("Timer 1") > 0.2
    assert timer_collection.duration("Timer 1") < 1.2

    # Test that multiple timers can be started and stopped
    timer_collection.reset("Timer 1")
    timer_collection.start("Timer 1")
    timer_collection.start("Timer 2")
    time.sleep(0.1)
    timer_collection.stop("Timer 1")
    time.sleep(1.0)
    timer_collection.stop("Timer 2")
    assert timer_collection.duration("Timer 1") > 0.1
    assert timer_collection.duration("Timer 1") < 1.0
    assert timer_collection.duration("Timer 2") > 1.1

    # Test that the timer collection can be converted to a string
    assert isinstance(str(timer_collection), str)

    # Test that the timer collection can return the timer names
    assert all((name in timer_collection.names() for name in ["Timer 1", "Timer 2"]))

    # Test that all timers can be reset
    timer_collection.reset_all()
    assert timer_collection.duration("Timer 1") == 0.0
    assert timer_collection.duration("Timer 2") == 0.0

    # Test that appropriate errors are raised when accessing non-existent timers
    with pytest.raises(KeyError):
        timer_collection.stop("Timer 3")
    with pytest.raises(KeyError):
        timer_collection.duration("Timer 3")
    with pytest.raises(KeyError):
        timer_collection.reset("Timer 3")
