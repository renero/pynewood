import pytest

from ..config import config


#############################################################################
# The tests

def test_config():
    params, logger = config("kk")
    assert len(params) >= 1, pytest.fail("No Configuration loaded")
