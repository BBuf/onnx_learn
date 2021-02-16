import torch
import pytest
import numpy as np
from torch import nn
from onnx.backend.test.case.node.pad import pad_impl

from onnx2pytorch.helpers import to_onnx
from onnx2pytorch.utils import (
    is_constant,
    get_selection,
    assign_values_to_dim,
    get_activation_value,
    extract_padding_params_for_conv_layer,
    extract_padding_params,
)


@pytest.fixture
def inp():
    return torch.rand(10, 10)


def test_is_constant():
    a = torch.tensor([1])
    assert is_constant(a)

    a = torch.tensor(1)
    assert is_constant(a)

    a = torch.tensor([1, 2])
    assert not is_constant(a)


def test_get_selection():
    indices = torch.tensor([1, 2, 5])
    with pytest.raises(AssertionError):
        get_selection(indices, -1)

    assert [indices] == get_selection(indices, 0)
    assert [slice(None), indices] == get_selection(indices, 1)


def test_get_selection_2():
    """Behaviour with python lists is unfortunately not working the same."""
    inp = torch.rand(3, 3, 3)
    indices = torch.tensor(0)

    selection = get_selection(indices, 0)
    assert torch.equal(inp[selection], inp[0])

    selection = get_selection(indices, 1)
    assert torch.equal(inp[selection], inp[:, 0])


@pytest.mark.parametrize(
    "val, dim, inplace", [[torch.zeros(4, 10), 0, False], [torch.zeros(10, 4), 1, True]]
)
def test_assign_values_to_dim(inp, val, dim, inplace):
    indices = torch.tensor([2, 4, 6, 8])

    out = inp.clone()
    if dim == 0:
        out[indices] = val
    elif dim == 1:
        out[:, indices] = val

    res = assign_values_to_dim(inp, val, indices, dim, inplace)
    if inplace:
        assert torch.equal(inp, out)
        assert torch.equal(res, out)
    else:
        # input should not be changed when inplace=False
        assert not torch.equal(inp, out)
        assert torch.equal(res, out)


def test_get_activation_value():
    inp = torch.ones(1, 1, 10, 10).numpy()
    model = nn.Sequential(nn.Conv2d(1, 3, 3), nn.Conv2d(3, 1, 3))
    model[0].weight.data *= 0
    model[0].weight.data += 1
    model.eval()

    onnx_model = to_onnx(model, inp.shape)

    activation_name = onnx_model.graph.node[0].output[0]
    value = get_activation_value(onnx_model, inp, activation_name)
    assert value[0].shape == (1, 3, 8, 8)
    a = value[0].round()
    b = 9 * np.ones((1, 3, 8, 8), dtype=np.float32)
    assert (a == b).all()


def test_get_activation_value_2():
    """Get multiple outputs from onnx model."""
    inp = torch.ones(1, 1, 10, 10).numpy()
    model = nn.Sequential(nn.Conv2d(1, 3, 3), nn.Conv2d(3, 1, 3))
    onnx_model = to_onnx(model, inp.shape)

    activation_names = [x.output[0] for x in onnx_model.graph.node]
    values = get_activation_value(onnx_model, inp, activation_names)
    assert values[0].shape == (1, 3, 8, 8)
    assert values[1].shape == (1, 1, 6, 6)


@pytest.mark.parametrize(
    "pads, output",
    [
        ([1, 1, 1, 1], [1, 1]),
        ([0, 0, 0, 0], [0, 0]),
        ([1, 0], nn.ConstantPad1d([1, 0], 0)),
        ([1, 2], nn.ConstantPad1d([1, 2], 0)),
        ([1, 1, 0, 0], nn.ConstantPad2d([1, 0, 1, 0], 0)),
        ([1, 1, 1, 0, 0, 0], nn.ConstantPad3d([1, 0, 1, 0, 1, 0], 0)),
    ],
)
def test_extract_padding_params_for_conv_layer(pads, output):
    out = extract_padding_params_for_conv_layer(pads)
    if isinstance(output, nn.Module):
        s = len(pads) // 2
        inp = np.random.rand(*s * [3])
        expected_out = pad_impl(inp, np.array(pads), "constant", 0)
        infered_out = out(torch.from_numpy(inp)).numpy()
        assert (expected_out == infered_out).all()
        assert output._get_name() == out._get_name()
        assert output.padding == out.padding
        assert output.value == out.value
    else:
        assert out == output


@pytest.fixture
def weight():
    return torch.rand(1, 3, 10, 10)


@pytest.mark.parametrize(
    "onnx_pads, torch_pads",
    [
        ([2, 2], [2, 2]),
        ([1, 2, 1, 2], [2, 2, 1, 1]),
        ([1, 2, 3, 4, 1, 2, 3, 4], [4, 4, 3, 3, 2, 2, 1, 1]),
        ([0, 0, 1, 2, 0, 0, 1, 2], [2, 2, 1, 1]),
    ],
)
def test_extract_padding_params(weight, onnx_pads, torch_pads):
    out_pads = extract_padding_params(onnx_pads)
    assert out_pads == torch_pads
