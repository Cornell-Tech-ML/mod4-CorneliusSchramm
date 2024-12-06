from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    # raise NotImplementedError("Need to implement for Task 4.3")
    new_h = height // kh
    new_w = width // kw

    input = input.contiguous()
    input = input.view(batch, channel, new_h, kh, new_w, kw)

    # do permutation
    input = input.permute(0, 1, 2, 4, 3, 5)

    input = input.contiguous()
    input = input.view(batch, channel, new_h, new_w, kw * kh)

    return input, new_h, new_w


# TODO: Implement for Task 4.3.
def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled average pooling 2D

    Args:
    ----
        input : batch x channel x height x width
        kernel : height x width of pooling

    Returns:
    -------
        Pooled tensor

    """
    batch, channel, height, width = input.shape
    input, h, w = tile(input, kernel)
    input = input.mean(4)
    input = input.view(batch, channel, h, w)
    return input


max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor.

    Args:
    ----
        input : input tensor
        dim : dimension to apply argmax

    Returns:
    -------
        A 1-hot tensor of the argmax

    """
    out = max_reduce(input, dim)
    return input == out


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Forward of max should be max reduction"""
        dim_number = int(dim.item())
        ctx.save_for_backward(input, dim)
        return max_reduce(input, dim_number)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward of max should be argmax (see above)"""
        input, dim = ctx.saved_values
        dim_number = int(dim.item())
        return grad_output * argmax(input, dim_number), 0.0


def max(input: Tensor, dim: int) -> Tensor:
    """Compute the max of a tensor over a given dimension."""
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    r"""Compute the softmax as a tensor.

        :math:`z_i = \frac{e^{x_i}}{\sum_i e^{x_i}}`

    Args:
    ----
        input : input tensor
        dim : dimension to apply softmax

    Returns:
    -------
        softmax tensor

    """
    input = input.exp()
    return input / input.sum(dim)


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    r"""Compute the log of the softmax as a tensor.

    Args:
    ----
        input : input tensor
        dim : dimension to apply logsoftmax

    Returns:
    -------
        logsoftmax tensor

    """
    return softmax(input, dim).log()


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled max pooling 2D

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor : pooled tensor

    """
    batch, channel, height, width = input.shape
    input, t_h, t_w = tile(input, kernel)
    input = max(input, 4)
    input = input.view(batch, channel, t_h, t_w)
    return input


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Dropout positions based on random noise.

    Args:
    ----
        input : input tensor
        rate : probability [0, 1) of dropping out each position
        ignore : skip dropout, i.e. do nothing at all

    Returns:
    -------
        tensor with randoom positions dropped out

    """
    if not ignore:
        b_tensor = rand(input.shape, input.backend) > rate
        input = b_tensor * input
    return input
