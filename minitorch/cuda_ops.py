# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs) -> Fn:  # noqa: ANN003
    """JIT compile a function for a specific device.
    This function uses the `_jit` decorator to compile the given function `fn`
    for execution on a specific device, such as a GPU. The `device` parameter
    is set to `True` to indicate that the function should be compiled for a
    device.

    Args:
    ----
        fn (Fn): The function to be JIT compiled.
        **kwargs: Additional keyword arguments to be passed to the `_jit` decorator.

    Returns:
    -------
        Fn: The JIT compiled function.

    """
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn, **kwargs) -> FakeCUDAKernel:  # noqa: ANN001, ANN003, D103
    """JIT compile a function for a specific device."""
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:  # noqa: D102
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(  # noqa: D102
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:  # noqa: D102
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement
def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)  # noqa: F841
        in_index = cuda.local.array(MAX_DIMS, numba.int32)  # noqa: F841
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x  # noqa: F841

        if i < out_size:
            # get the index of i in out_shape
            to_index(i, out_shape, out_index)
            # broadcast the index
            broadcast_index(out_index, out_shape, in_shape, in_index)
            o = index_to_position(out_index, out_strides)
            j = index_to_position(in_index, in_strides)
            out[o] = fn(in_storage[j])

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)  # noqa: F841
        a_index = cuda.local.array(MAX_DIMS, numba.int32)  # noqa: F841
        b_index = cuda.local.array(MAX_DIMS, numba.int32)  # noqa: F841
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x  # noqa: F841
        if i < out_size:
            # get the index of i in out_shape
            to_index(i, out_shape, out_index)
            # broadcast
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            # get the position
            o = index_to_position(out_index, out_strides)
            j = index_to_position(a_index, a_strides)
            k = index_to_position(b_index, b_strides)
            # write it to out
            out[o] = fn(a_storage[j], b_storage[k])

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    r"""Here This is a practice sum kernel to prepare for reduce.

    Given an array of length $n$ and out of size $n // \text{blockDIM}$
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """
    BLOCK_DIM = 32

    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x

    # handle edge cases when input size isn't perfectly divisible by BLOCK_DIM:
    if i < size:
        cache[pos] = a[i]
    else:
        cache[pos] = 0
    cuda.syncthreads()
    # Now reduce with stride pattern
    stride = BLOCK_DIM // 2
    while stride > 0:
        if pos < stride:
            cache[pos] += cache[pos + stride]
        cuda.syncthreads()
        stride //= 2
    # write cache[0] to global memory
    if pos == 0:
        out[cuda.blockIdx.x] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    """Practice sum function to prepare for reduce."""
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)  # noqa: F841
        out_index = cuda.local.array(MAX_DIMS, numba.int32)  # noqa: F841
        out_pos = cuda.blockIdx.x  # noqa: F841
        pos = cuda.threadIdx.x  # noqa: F841k
        # i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        size = a_shape[reduce_dim]

        # 1. Figure out which output position this block is handling
        to_index(out_pos, out_shape, out_index)

        # 2. Load the correct slice into shared memory
        out_index[reduce_dim] = pos  # Only vary along reduce_dim
        if pos < size:
            j = index_to_position(out_index, a_strides)
            cache[pos] = a_storage[j]
        else:
            cache[pos] = reduce_value
        cuda.syncthreads()

        # Then: Parallel reduction within shared memory
        stride = BLOCK_DIM // 2
        while stride > 0:
            if pos < stride:
                cache[pos] = fn(cache[pos], cache[pos + stride])
            cuda.syncthreads()
            stride //= 2
        # write cache[0] to global memory
        if pos == 0:
            out[out_pos] = cache[0]

    return jit(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """Practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square

    """
    BLOCK_DIM = 32  # noqa: F841
    # # Details regarding this thread
    # block_y = cuda.blockIdx.y  # Row
    # block_x = cuda.blockIdx.x  # Column

    # # Details regarding this thread
    # thread_y = cuda.threadIdx.y  # Row
    # thread_x = cuda.threadIdx.x  # Column

    # # Working on Out[out_y, out_x]
    # out_y = block_y * BLOCK_DIM + thread_y
    # out_x = block_x * BLOCK_DIM + thread_x

    # # Shared memory for A and B
    # a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    # b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # # Load tiles into shared memory
    # # for A:
    # #   row_index is gloabl out_y,
    # #   col_index is tile_number * BLOCK_DIM + thread_x
    # # for B:
    # #   row_index is tile number * BLOCK_DIM + thread_y
    # #   col_index is global out_x
    # value = 0.0
    # for tile in range(size // BLOCK_DIM):
    #     # row major layout -> in linear form ordinal is (matrix width  * global_row_index) + col_index
    #     a_shared[thread_y, thread_x] = a[out_y * size + tile * BLOCK_DIM + thread_x]
    #     b_shared[thread_y, thread_x] = b[tile * BLOCK_DIM + thread_y + out_x]
    #     # Synchronize threads
    #     cuda.syncthreads()
    #     # Compute partial sum dot product A rows * B cols
    #     for k in range(BLOCK_DIM):
    #         value += a_shared[thread_y, k] * b_shared[k, thread_x]
    #         # Synchronize threads
    #     cuda.syncthreads()
    # # Write to global memory
    # out[out_y * size + out_x] = value
    share_a = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    share_b = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    i = cuda.threadIdx.x
    j = cuda.threadIdx.y

    if i >= size or j >= size:
        return

    # move to share memory
    share_a[i, j] = a[size * i + j]
    share_b[i, j] = b[size * i + j]
    cuda.syncthreads()

    # accumulate a[i,k]*b[k,j]
    acc = 0.0
    for k in range(size):
        acc += share_a[i, k] * share_b[k, j]

    # write to out
    out[size * i + j] = acc


jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    """Practice matrix multiply function to prepare for matmul."""
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    # Calculate batch strides, accounting for broadcasting (stride=0 if dim size=1)
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    # Get current batch index from block z-coordinate
    batch = cuda.blockIdx.z

    # Define tile size for shared memory blocking
    BLOCK_DIM = 32
    # Allocate shared memory for tile of matrix A and B
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # Calculate global thread indices (position in output matrix)
    # i = row index, j = column index
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x  # Global row index
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y  # Global column index

    # Local thread indices within the block (position in shared memory)
    pi = cuda.threadIdx.x  # Local row index
    pj = cuda.threadIdx.y  # Local column index

    # Initialize accumulator for dot product
    acc = 0.0

    # Iterate over tiles of size BLOCK_DIM in the K dimension
    for phase in range(0, a_shape[-1], BLOCK_DIM):
        # Load tile from matrix A into shared memory
        # Each thread loads one element
        if phase + pj < a_shape[-1] and i < a_shape[-2]:
            # Calculate global memory position using strides
            a_shared[pi, pj] = a_storage[
                batch * a_batch_stride  # Batch offset
                + i * a_strides[1]  # Row offset
                + (phase + pj) * a_strides[2]  # Column offset
            ]
        else:
            # Pad with zeros if outside matrix bounds
            a_shared[pi, pj] = 0.0

        # Load tile from matrix B into shared memory
        if phase + pi < b_shape[-2] and j < b_shape[-1]:
            # Calculate global memory position using strides
            b_shared[pi, pj] = b_storage[
                batch * b_batch_stride  # Batch offset
                + (phase + pi) * b_strides[1]  # Row offset
                + j * b_strides[2]  # Column offset
            ]
        else:
            # Pad with zeros if outside matrix bounds
            b_shared[pi, pj] = 0.0

        # Ensure all threads have loaded their data before computation
        cuda.syncthreads()

        # Compute partial dot product using the current tile
        if i < out_shape[1] and j < out_shape[2]:
            # Multiply and accumulate along K dimension
            for k in range(BLOCK_DIM):
                acc += a_shared[pi, k] * b_shared[k, pj]

        # Ensure all threads complete computation before loading next tile
        cuda.syncthreads()

    # Calculate output position using strides
    o = batch * out_strides[0] + i * out_strides[1] + j * out_strides[2]

    # Write result to global memory if within bounds
    if o < out_size and i < out_shape[1] and j < out_shape[2]:
        out[o] = acc


tensor_matrix_multiply = jit(_tensor_matrix_multiply)
