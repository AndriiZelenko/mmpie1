"""Unit test methods for the mmpie1.core.config.Config class."""
import os

import pytest

from mmpie1 import Config
from mmpie1_setup import initial_mmpie1_setup


def test_functionality():
    config = Config()
    assert isinstance(str(config), str)
    assert isinstance(config.pretty_print(), str)
    assert isinstance(config.as_dict(), dict)


def test_configuration():
    initial_mmpie1_setup()
    config = Config()
    dirs = (
        config["DIR_PATHS"]["ROOT"],
        config["DIR_PATHS"]["DATA"],
        config["DIR_PATHS"]["LOGS"],
        config["DIR_PATHS"]["TENSORBOARD"],
        config["DIR_PATHS"]["MLFLOW"],
        config["DIR_PATHS"]["TEMP"],
    )
    for dir_ in dirs:
        assert os.path.isdir(dir_)


def test_config_not_found():
    with pytest.raises(Exception):
        config = Config("/this/config/path/does/not/exist/config.ini")
        isinstance(config, Config)
