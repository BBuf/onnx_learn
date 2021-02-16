import torch
import pytest

from onnx2pytorch.operations import Reshape


@pytest.fixture
def inp():
    return torch.rand(35, 1, 200)


@pytest.fixture
def pruned_inp():
    return torch.rand(35, 1, 160)


def test_reshape(inp, pruned_inp):
    """Pass shape in forward."""
    op = Reshape()
    shape = torch.Size((35, 2, 100))
    out = op(inp, shape)
    assert out.shape == shape

    # with the same input, the output shape should not change
    out = op(inp, shape)
    assert out.shape == shape

    # if input changes due to pruning, reshape should work
    # and output shape should change accordingly
    expected_shape = torch.Size((35, 2, 80))
    out = op(pruned_inp, shape)
    assert out.shape == expected_shape


def test_reshape_2(inp, pruned_inp):
    """Pass shape in init."""
    shape = torch.Size((35, 2, 100))
    op = Reshape(shape)
    out = op(inp)
    assert out.shape == shape

    # input changes due to pruning, reshape should work
    expected_shape = torch.Size((35, 2, 80))
    out = op(pruned_inp)
    assert out.shape == expected_shape
