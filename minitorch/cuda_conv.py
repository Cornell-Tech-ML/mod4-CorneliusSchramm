from typing import Tuple

import numba
from numba import cuda

from .autodiff import Context
from .tensor import Tensor
from .tensor_data import (
    Shape,
    Storage,
    Strides,
    UserShape,
    to_index,
    index_to_position,
    broadcast_index,
)
from .tensor_functions import Function

to_index = cuda.jit(device=True)(to_index)
index_to_position = cuda.jit(device=True)(index_to_position)
broadcast_index = cuda.jit(device=True)(broadcast_index)


def _tensor_conv1d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """Perform a 1D convolution operation.

    Args:
    ----
        out: The output storage.
        out_shape: The shape of the output tensor.
        out_strides: The strides of the output tensor.
        out_size: The total size of the output tensor.
        input: The input storage.
        input_shape: The shape of the input tensor.
        input_strides: The strides of the input tensor.
        weight: The weight storage.
        weight_shape: The shape of the weight tensor.
        weight_strides: The strides of the weight tensor.
        reverse: Whether to anchor the weight at the left (False) or right (True).

    Returns:
    -------
        None

    """
    BLOCK_DIM = 16
    BLOCK_DIM2 = 32
    batch_, out_channels, out_width = out_shape
    batch, in_channels, width = input_shape
    out_channels_, in_channels_, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )
    assert out_width <= width

    out_w_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x  # type: ignore
    out_w_cache_start = cuda.blockIdx.x * cuda.blockDim.x  # type: ignore
    out_ch_idx = cuda.blockIdx.z  # type: ignore
    th_x = cuda.threadIdx.x  # type: ignore
    th_y = cuda.threadIdx.y  # type: ignore

    shared_weight_cache = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)  # type: ignore
    shared_input_cache = cuda.shared.array((BLOCK_DIM, BLOCK_DIM2), numba.float64)  # type: ignore
    ws0, ws1, ws2 = weight_strides
    is0, is1, is2 = input_strides
    os0, os1, os2 = out_strides

    kernel_w_dir = -1 if reverse else 1

    for batch_i in range(batch):
        accum_val = 0.0
        for in_ch_start in range(0, in_channels, BLOCK_DIM):
            in_ch_cache_pos = in_ch_start + th_x  # type: ignore
            for kernel_w_start in range(0, kw, BLOCK_DIM):
                kernel_w_cur = kernel_w_start + th_y  # type: ignore
                if in_ch_cache_pos < in_channels and kernel_w_cur < kw:
                    weight_mem_pos = (
                        out_ch_idx * ws0 + in_ch_cache_pos * ws1 + kernel_w_cur * ws2
                    )
                    shared_weight_cache[(th_x, th_y)] = weight[weight_mem_pos]  # type: ignore
                else:
                    shared_weight_cache[(th_x, th_y)] = 0.0  # type: ignore
                numba.cuda.syncthreads()  # type: ignore

                for cache_w_bias in range(0, BLOCK_DIM2, BLOCK_DIM):
                    if reverse:
                        w_cache_pos = (
                            out_w_cache_start
                            - kernel_w_start
                            - BLOCK_DIM
                            + 1
                            + cache_w_bias
                            + th_y
                        )
                    else:
                        w_cache_pos = (
                            out_w_cache_start + kernel_w_start + cache_w_bias + th_y
                        )
                    if in_ch_cache_pos < in_channels and 0 <= w_cache_pos < width:
                        input_mem_pos = (
                            batch_i * is0 + in_ch_cache_pos * is1 + w_cache_pos * is2
                        )
                        shared_input_cache[(th_x, cache_w_bias + th_y)] = input[  # type: ignore
                            input_mem_pos
                        ]  # type: ignore
                    else:
                        shared_input_cache[(th_x, cache_w_bias + th_y)] = 0.0  # type: ignore
                numba.cuda.syncthreads()  # type: ignore

                if th_y == 0 and out_w_idx < out_width:
                    for in_channel_i in range(
                        in_ch_start, min(in_channels, in_ch_start + BLOCK_DIM)
                    ):
                        for kwi in range(
                            kernel_w_start, min(kw, kernel_w_start + BLOCK_DIM)
                        ):
                            cur_w = out_w_idx + kwi * kernel_w_dir
                            if reverse:
                                w_cache_min = (
                                    out_w_cache_start - kernel_w_start - BLOCK_DIM + 1
                                )
                            else:
                                w_cache_min = out_w_cache_start + kernel_w_start
                            w_cache_max = w_cache_min + BLOCK_DIM2
                            if (
                                w_cache_min <= cur_w < w_cache_max
                                and 0 <= cur_w < width
                            ):
                                accum_val += (
                                    shared_weight_cache[  # type: ignore
                                        (
                                            in_channel_i - in_ch_start,
                                            kwi - kernel_w_start,
                                        )
                                    ]
                                    * shared_input_cache[  # type: ignore
                                        (
                                            in_channel_i - in_ch_start,
                                            abs(cur_w - w_cache_min),
                                        )
                                    ]
                                )
                numba.cuda.syncthreads()  # type: ignore

        if th_y == 0 and out_w_idx < out_width:
            out_pos = batch_i * os0 + out_ch_idx * os1 + out_w_idx * os2
            out[out_pos] = accum_val


tensor_conv1d = cuda.jit()(_tensor_conv1d)


class Conv1dFun(Function):
    @staticmethod
    def forward_inner(
        output_shape: UserShape, input: Tensor, weight: Tensor, reversed: bool = False
    ) -> Tensor:
        """Compute a 1D Convolution.

        Args:
        ----
            output_shape: The desired output shape.
            input: The input tensor with shape (batch, in_channels, width).
            weight: The weight tensor with shape (out_channels, in_channels, kernel_width).
            reversed: Whether the weight is anchored at the left (False) or right (True).

        Returns:
        -------
            Tensor: The convolved output tensor with shape (batch, out_channels, width).

        """
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        output = input.zeros(output_shape)
        THREADS_PER_BLOCK = 16
        blocks_per_grid = (
            (w + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            1,
            out_channels,
        )
        threads_per_block = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_conv1d[blocks_per_grid, threads_per_block](  # type: ignore
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), reversed
        )
        return output

    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Forward pass for the 1D Convolution.

        Args:
        ----
            ctx: The context object used to save intermediate values for backward pass.
            input: The input tensor of shape (batch, in_channels, width).
            weight: The weight tensor of shape (out_channels, in_channels, kernel_width).

        Returns:
        -------
            Tensor: The output tensor of shape (batch, out_channels, width).

        """
        ctx.save_for_backward(input, weight)
        output = Conv1dFun.forward_inner(
            (input.shape[0], weight.shape[0], input.shape[2]),
            input,
            weight,
            reversed=False,
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for the 1D Convolution.

        Args:
        ----
            ctx: Context object with saved forward pass tensors.
            grad_output: The gradient of the loss with respect to the output tensor.

        Returns:
        -------
            Tuple[Tensor, Tensor]: A tuple containing the gradients with respect to input and weight.

        """
        input, weight = ctx.saved_values
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        grad_weight = Conv1dFun.forward_inner(
            (weight.shape[1], weight.shape[0], weight.shape[2]),
            new_input,
            new_grad_output,
            reversed=False,
        )
        grad_weight = grad_weight.permute(1, 0, 2)

        new_weight = weight.permute(1, 0, 2)
        grad_input = Conv1dFun.forward_inner(
            input.shape, grad_output, new_weight, reversed=True
        )
        return grad_input, grad_weight


conv1d = Conv1dFun.apply


def _tensor_conv2d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """Perform a 2D convolution operation.

    Args:
    ----
        out: The output storage.
        out_shape: The shape of the output tensor.
        out_strides: The strides of the output tensor.
        out_size: The total size of the output tensor.
        input: The input storage.
        input_shape: The shape of the input tensor.
        input_strides: The strides of the input tensor.
        weight: The weight storage.
        weight_shape: The shape of the weight tensor.
        weight_strides: The strides of the weight tensor.
        reverse: Whether to anchor the weight top-left (False) or bottom-right (True).

    Returns:
    -------
        None

    """
    batch_, out_channels, out_height, out_width = out_shape
    batch, in_channels, height, width = input_shape
    out_channels_, in_channels_, kh, kw = weight_shape

    BLOCK_DIM = 16
    BLOCK_DIM2 = 32

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )
    assert out_width <= width and out_height <= height

    out_w_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x  # type: ignore
    out_h_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y  # type: ignore
    out_w_cache_start = cuda.blockIdx.x * cuda.blockDim.x  # type: ignore
    out_h_cache_start = cuda.blockIdx.y * cuda.blockDim.y  # type: ignore
    out_ch_idx = cuda.blockIdx.z  # type: ignore
    th_x = cuda.threadIdx.x  # type: ignore
    th_y = cuda.threadIdx.y  # type: ignore

    shared_weight_cache = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)  # type: ignore
    shared_input_cache = cuda.shared.array((BLOCK_DIM2, BLOCK_DIM2), numba.float64)  # type: ignore
    ws0, ws1, ws2, ws3 = weight_strides
    is0, is1, is2, is3 = input_strides
    os0, os1, os2, os3 = out_strides

    kernel_dir = -1 if reverse else 1

    for batch_i in range(batch):
        out_pos = batch_i * os0 + out_ch_idx * os1 + out_h_idx * os2 + out_w_idx * os3
        accum_val = 0.0
        for in_channel_i in range(in_channels):
            for kernel_h_start in range(0, kh, BLOCK_DIM):
                for kernel_w_start in range(0, kw, BLOCK_DIM):
                    kernel_w_cur = kernel_w_start + th_x  # type: ignore
                    kernel_h_cur = kernel_h_start + th_y  # type: ignore

                    if kernel_h_cur < kh and kernel_w_cur < kw:
                        weight_mem_pos = (
                            out_ch_idx * ws0
                            + in_channel_i * ws1
                            + kernel_h_cur * ws2
                            + kernel_w_cur * ws3
                        )
                        shared_weight_cache[(th_x, th_y)] = weight[weight_mem_pos]  # type: ignore
                    else:
                        shared_weight_cache[(th_x, th_y)] = 0.0  # type: ignore
                    numba.cuda.syncthreads()  # type: ignore

                    for cache_w_bias in range(0, BLOCK_DIM2, BLOCK_DIM):
                        for cache_h_bias in range(0, BLOCK_DIM2, BLOCK_DIM):
                            if reverse:
                                w_cache_pos = (
                                    out_w_cache_start
                                    - kernel_w_start
                                    - BLOCK_DIM
                                    + 1
                                    + cache_w_bias
                                    + th_x
                                )
                                h_cache_pos = (
                                    out_h_cache_start
                                    - kernel_h_start
                                    - BLOCK_DIM
                                    + 1
                                    + cache_h_bias
                                    + th_y
                                )
                            else:
                                w_cache_pos = (
                                    out_w_cache_start
                                    + kernel_w_start
                                    + cache_w_bias
                                    + th_x
                                )
                                h_cache_pos = (
                                    out_h_cache_start
                                    + kernel_h_start
                                    + cache_h_bias
                                    + th_y
                                )

                            if 0 <= w_cache_pos < width and 0 <= h_cache_pos < height:
                                input_mem_pos = (
                                    batch_i * is0
                                    + in_channel_i * is1
                                    + h_cache_pos * is2
                                    + w_cache_pos * is3
                                )
                                shared_input_cache[  # type: ignore
                                    (cache_w_bias + th_x, cache_h_bias + th_y)  # type: ignore
                                ] = input[input_mem_pos]  # type: ignore
                            else:
                                shared_input_cache[  # type: ignore
                                    (cache_w_bias + th_x, cache_h_bias + th_y)  # type: ignore
                                ] = 0.0  # type: ignore
                            numba.cuda.syncthreads()  # type: ignore

                    if out_h_idx < out_height and out_w_idx < out_width:
                        for kernel_h_idx in range(
                            kernel_h_start, min(kh, kernel_h_start + BLOCK_DIM)
                        ):
                            cur_h = out_h_idx + kernel_h_idx * kernel_dir
                            if reverse:
                                h_cache_min = (
                                    out_h_cache_start - kernel_h_start - BLOCK_DIM + 1
                                )
                            else:
                                h_cache_min = out_h_cache_start + kernel_h_start
                            h_cache_max = h_cache_min + BLOCK_DIM2
                            if not (
                                0 <= cur_h < height
                                and h_cache_min <= cur_h < h_cache_max
                            ):
                                continue

                            for kernel_w_idx in range(
                                kernel_w_start, min(kw, kernel_w_start + BLOCK_DIM)
                            ):
                                cur_w = out_w_idx + kernel_w_idx * kernel_dir
                                if reverse:
                                    w_cache_min = (
                                        out_w_cache_start
                                        - kernel_w_start
                                        - BLOCK_DIM
                                        + 1
                                    )
                                else:
                                    w_cache_min = out_w_cache_start + kernel_w_start
                                w_cache_max = w_cache_min + BLOCK_DIM2
                                if not (
                                    0 <= cur_w < width
                                    and w_cache_min <= cur_w < w_cache_max
                                ):
                                    continue
                                accum_val += (
                                    shared_weight_cache[  # type: ignore
                                        (
                                            kernel_w_idx - kernel_w_start,
                                            kernel_h_idx - kernel_h_start,
                                        )
                                    ]  # type: ignore
                                    * shared_input_cache[  # type: ignore
                                        (
                                            abs(cur_w - w_cache_min),
                                            abs(cur_h - h_cache_min),
                                        )
                                    ]
                                )
                    numba.cuda.syncthreads()  # type: ignore

        if out_h_idx < out_height and out_w_idx < out_width:
            out[out_pos] = accum_val


tensor_conv2d = cuda.jit()(_tensor_conv2d)


class Conv2dFun(Function):
    @staticmethod
    def forward_inner(
        output_shape: UserShape, input: Tensor, weight: Tensor, reversed: bool = False
    ) -> Tensor:
        """Compute a 2D Convolution.

        Args:
        ----
            output_shape: The desired output shape.
            input: The input tensor with shape (batch, in_channels, height, width).
            weight: The weight tensor with shape (out_channels, in_channels, kernel_height, kernel_width).
            reversed: Whether the weight is anchored top-left (False) or bottom-right (True).

        Returns:
        -------
            Tensor: The convolved output tensor with shape (batch, out_channels, height, width).

        """
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2

        output = input.zeros(output_shape)
        THREADS_PER_BLOCK = 16
        blocks_per_grid = (
            (w + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (h + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out_channels,
        )
        threads_per_block = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_conv2d[blocks_per_grid, threads_per_block](  # type: ignore
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), reversed
        )
        return output

    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Forward pass for the 2D Convolution.

        Args:
        ----
            ctx: The context object for backward computation.
            input: The input tensor of shape (batch, in_channels, height, width).
            weight: The weight tensor of shape (out_channels, in_channels, kernel_height, kernel_width).

        Returns:
        -------
            Tensor: The output tensor of shape (batch, out_channels, height, width).

        """
        ctx.save_for_backward(input, weight)
        output = Conv2dFun.forward_inner(
            (input.shape[0], weight.shape[0], input.shape[2], input.shape[3]),
            input,
            weight,
            reversed=False,
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for the 2D Convolution.

        Args:
        ----
            ctx: Context object with forward pass saved values.
            grad_output: Gradient of the loss with respect to the output tensor.

        Returns:
        -------
            Tuple[Tensor, Tensor]: A tuple containing gradients with respect to the input and weight tensors.

        """
        input, weight = ctx.saved_values

        new_input = input.permute(1, 0, 2, 3)
        new_grad_output = grad_output.permute(1, 0, 2, 3)
        grad_weight = Conv2dFun.forward_inner(
            (weight.shape[1], weight.shape[0], weight.shape[2], weight.shape[3]),
            new_input,
            new_grad_output,
            reversed=False,
        )
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        new_weight = weight.permute(1, 0, 2, 3)
        grad_input = Conv2dFun.forward_inner(
            input.shape, grad_output, new_weight, reversed=True
        )
        return grad_input, grad_weight


conv2d = Conv2dFun.apply
