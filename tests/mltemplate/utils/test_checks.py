"""Unit test methods for mmpie1.utils.checks utility module."""
from mmpie1.utils import ifnone


def test_ifnone():
    assert ifnone(val=True, default=False) is True
    assert ifnone(val=None, default=False) is False

    assert ifnone(val=5, default=10) == 5
    assert ifnone(val=None, default=10) == 10
