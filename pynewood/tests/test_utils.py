import pytest

from ..utils import reset_seeds, gen_toy_dataset


#############################################################################
# The tests


def test_reset_seeds():
    try:
        reset_seeds(1)
    except AttributeError:
        pytest.fail('Unexpected error')


def test_gen_toy_dataset():
    try:
        data, true_structure = gen_toy_dataset(scale=True)
    except ValueError:
        pytest.fail('Unexpected error')

