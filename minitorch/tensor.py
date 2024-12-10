"""Implementation of the core Tensor object for autodifferentiation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import operators
from .autodiff import Context, Variable, backpropagate
from .tensor_data import TensorData

# Comment these out if not yet implemented
from .tensor_functions import (
    Copy,
    Inv,
    MatMul,
    Mul,
    Add,
    Neg,
    LT,
    EQ,
    All,
    Sigmoid,
    ReLU,
    Log,
    Exp,
    Sum,
    IsClose,
    Permute,
    View,
    tensor,
)

if TYPE_CHECKING:
    from typing import Any, Iterable, List, Optional, Sequence, Tuple, Type, Union

    import numpy.typing as npt

    from .tensor_data import Shape, Storage, Strides, UserIndex, UserShape, UserStrides
    from .tensor_functions import Function
    from .tensor_ops import TensorBackend

    TensorLike = Union[float, int, "Tensor"]


@dataclass
class History:
    """`History` stores the history of `Function` operations that was
    used to construct the current Variable.
    """

    last_fn: Optional[Type[Function]] = None
    ctx: Optional[Context] = None
    inputs: Sequence["Tensor"] = ()  # type: ignore


_tensor_count = 0


class Tensor:
    """Tensor is a generalization of Scalar in that it is a Variable that
    handles multidimensional arrays.
    """

    backend: TensorBackend
    history: Optional[History]
    grad: Optional[Tensor]
    _tensor: TensorData
    unique_id: int
    name: str

    def __init__(
        self,
        v: TensorData,
        back: Optional[History] = None,
        name: Optional[str] = None,
        backend: Optional[TensorBackend] = None,
    ):
        global _tensor_count
        _tensor_count += 1
        self.unique_id = _tensor_count
        assert isinstance(v, TensorData)
        assert backend is not None
        self._tensor = v
        self.history = back
        self.backend = backend
        self.grad = None
        if name is not None:
            self.name = name
        else:
            self.name = str(self.unique_id)

        self.f = backend
        # self.size = 1
        # for dim in self.shape:
        #     self.size *= dim

    def zero_grad_(self) -> None:
        """Resets the gradient of the tensor to None."""
        self.grad = None

    def __hash__(self):
        return id(self)

    def requires_grad_(self, x: bool) -> None:
        """Set the requires_grad flag on the tensor."""
        self.history = History()

    def requires_grad(self) -> bool:
        """Returns whether the tensor requires gradients."""
        return self.history is not None

    def to_numpy(self) -> npt.NDArray[np.float64]:
        """Returns
        Converted to numpy array

        """
        return self.contiguous()._tensor._storage.reshape(self.shape)

    # Properties
    @property
    def shape(self) -> UserShape:
        """Returns:
        shape of the tensor
        """
        return self._tensor.shape

    @property
    def size(self) -> int:
        """Returns:
        int : size of the tensor
        """
        return self._tensor.size

    @property
    def dims(self) -> int:
        """Returns:
        int : dimensionality of the tensor
        """
        return self._tensor.dims

    def _ensure_tensor(self, b: TensorLike) -> Tensor:
        """Turns a python number into a tensor with the same backend."""
        if isinstance(b, (int, float)):
            c = Tensor.make([b], (1,), backend=self.backend)  # type: ignore
            # c = Tensor.make([b], cast(Sequence[int], (1,)), backend=self.backend) # leads to 2.3 FAILED tests/test_tensor.py::test_reduce_forward_all_dims - NameError: name 'Sequence' is not defined
            # c = Tensor.make([b], [1], backend=self.backend)  # Convert tuple to list
            # c = Tensor.make([b], (int(1),), backend=self.backend) # Argument of type "tuple[int]" cannot be assigned to parameter "shape" of type "UserShape" in function "make"  "tuple[int]" is not assignable to "Sequence[int]"

        else:
            b._type_(self.backend)
            c = b
        return c

    def item(self) -> float:
        """Convert a 1-element tensor to a float"""
        assert self.size == 1
        x: float = self._tensor._storage[0]
        return x

    def contiguous(self) -> Tensor:
        """Return a contiguous tensor with the same data"""
        return Copy.apply(self)

    def __repr__(self) -> str:
        return self._tensor.to_string()

    # def __getitem__(self, key: Union[int, UserIndex]) -> float:
    #     key2 = (key,) if isinstance(key, int) else key
    #     return self._tensor.get(key2)

    # def __setitem__(self, key: Union[int, UserIndex], val: float) -> None:
    #     key2 = (key,) if isinstance(key, int) else key
    #     self._tensor.set(key2, val)

    def __getitem__(self, key: UserIndex) -> float:
        key2: UserIndex = (key,) if isinstance(key, int) else key  # type: ignore
        return self._tensor.get(key2)

    def __setitem__(self, key: UserIndex, val: float) -> None:
        key2: UserIndex = (key,) if isinstance(key, int) else key  # type: ignore
        self._tensor.set(key2, val)

    # Internal methods used for autodiff.
    def _type_(self, backend: TensorBackend) -> None:
        self.backend = backend
        if backend.cuda:  # pragma: no cover
            self._tensor.to_cuda_()

    def _new(self, tensor_data: TensorData) -> Tensor:
        return Tensor(tensor_data, backend=self.backend)

    @staticmethod
    def make(
        storage: Union[Storage, List[float]],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
        backend: Optional[TensorBackend] = None,
    ) -> Tensor:
        """Create a new tensor from data"""
        return Tensor(TensorData(storage, shape, strides), backend=backend)

    def expand(self, other: Tensor) -> Tensor:
        """Method used to allow for backprop over broadcasting.

        This method is called when the output of `backward`
        is a different size than the input of `forward`.

        Args:
        ----
            other : backward tensor (must broadcast with self)

        Returns:
        -------
            Expanded version of `other` with the right derivatives

        """
        # Case 1: Both the same shape.
        if self.shape == other.shape:
            return other

        # Case 2: Backward is a smaller than self. Broadcast up.
        true_shape = TensorData.shape_broadcast(self.shape, other.shape)
        buf = self.zeros(true_shape)
        self.backend.id_map(other, buf)
        if self.shape == true_shape:
            return buf

        # Case 3: Still different, reduce extra dims.
        out = buf
        orig_shape = [1] * (len(out.shape) - len(self.shape)) + list(self.shape)
        for dim, shape in enumerate(out.shape):
            if orig_shape[dim] == 1 and shape != 1:
                out = self.backend.add_reduce(out, dim)
        assert out.size == self.size, f"{out.shape} {self.shape}"
        # START CODE CHANGE (2021)
        return Tensor.make(out._tensor._storage, self.shape, backend=self.backend)
        # END CODE CHANGE (2021)

    def zeros(self, shape: Optional[UserShape] = None) -> Tensor:
        """Create a tensor filled with zeros.
        This method generates a tensor of the specified shape, filled with zeros.
        If no shape is provided, it uses the shape of the current tensor.

        Args:
        ----
            shape (Optional[UserShape]): The shape of the tensor to create. If None,
                                         the shape of the current tensor is used.

        Returns:
        -------
            Tensor: A tensor filled with zeros of the specified shape.

        """

        def zero(shape: UserShape) -> Tensor:
            """Creates a tensor filled with zeros of the specified shape.

            Args:
            ----
                shape (UserShape): The shape of the tensor to create.

            Returns:
            -------
                Tensor: A tensor filled with zeros of the specified shape.

            """
            return Tensor.make(
                [0.0] * int(operators.prod(list(shape))), shape, backend=self.backend
            )

        if shape is None:
            out = zero(self.shape)
        else:
            out = zero(shape)
        out._type_(self.backend)
        return out

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        """Get the tensor data info as a tuple."""
        return self._tensor.tuple()

    def detach(self) -> Tensor:
        """Detach from backprop"""
        return Tensor(self._tensor, backend=self.backend)

    # Variable elements for backprop

    def accumulate_derivative(self, x: Any) -> None:
        """Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
        ----
            x : value to be accumulated

        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.grad is None:
            self.grad = Tensor.make(
                [0.0] * int(operators.prod(self.shape)),  # type: ignore
                self.shape,
                backend=self.backend,
            )
        self.grad += x

    def is_leaf(self) -> bool:
        """True if this variable created by the user (no `last_fn`)"""
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """Check if the tensor is constant (i.e., has no history)."""
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        """Returns the parent variables of the current tensor."""
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Compute the chain rule for backpropagation.

        This method calculates the gradients of the inputs with respect to the output
        using the chain rule of calculus. It leverages the stored history of operations
        to perform the backward pass.

        Args:
        ----
            d_output : The gradient of the output with respect to some loss.

        Returns:
        -------
            Iterable[Tuple[Variable, Any]: An iterable of tuples where each tuple contains an input variable and its
            corresponding gradient.

        """
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        x = h.last_fn._backward(h.ctx, d_output)
        assert len(x) == len(h.inputs), f"Bug in function {h.last_fn}"
        return [
            (inp, inp.expand(self._ensure_tensor(d_in)))
            for inp, d_in in zip(h.inputs, x)
        ]

    def backward(self, grad_output: Optional[Tensor] = None) -> None:
        """Performs backpropagation on the tensor.

        This method initiates the backpropagation process by calling the `backpropagate` function.
        If `grad_output` is not provided, it is assumed that the tensor is a scalar and a default
        gradient of 1.0 is used.

        Args:
        ----
            grad_output: Optional[Tensor] = None
                The gradient of the output with respect to some loss. If not provided, a default
                gradient is used for scalar tensors.

        """
        if grad_output is None:
            assert self.shape == (1,), "Must provide grad_output if non-scalar"
            grad_output = Tensor.make([1.0], (1,), backend=self.backend)  # type: ignore
        # import pdb; pdb.set_trace()
        backpropagate(self, grad_output)

    def __truediv__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self, Inv.apply(self._ensure_tensor(b)))

    def __rtruediv__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self._ensure_tensor(b), Inv.apply(self))

    def __matmul__(self, b: Tensor) -> Tensor:
        """Not used until Module 3"""
        return MatMul.apply(self, b)

    # @property
    # def shape(self) -> UserShape:
    #     """Returns
    #     shape of the tensor

    #     """
    #     return self._tensor.shape

    # Functions
    # TODO: Implement for Task 2.3.

    def __add__(self, other: TensorLike) -> Tensor:
        other = self._ensure_tensor(other)
        return Add.apply(self, other)

    def __radd__(self, other: TensorLike) -> Tensor:
        return self.__add__(other)

    # Add this method to the Tensor class
    def __mul__(self, other: Tensor) -> Tensor:
        other = self._ensure_tensor(other)
        return Mul.apply(self, other)

    def __rmul__(self, other: Tensor) -> Tensor:
        other = self._ensure_tensor(other)
        return Mul.apply(other, self)

    def __sub__(self, other: Tensor) -> Tensor:
        other = self._ensure_tensor(other)
        return Add.apply(self, Neg.apply(other))

    def __rsub__(self, other: Tensor) -> Tensor:
        other = self._ensure_tensor(other)
        return Add.apply(other, Neg.apply(self))

    def __neg__(self) -> Tensor:
        return Neg.apply(self)

    def __lt__(self, other: Tensor) -> Tensor:
        other = self._ensure_tensor(other)
        return LT.apply(self, other)

    def __gt__(self, b: TensorLike) -> Tensor:
        return LT.apply(self._ensure_tensor(b), self)

    def __eq__(self, other: Tensor) -> Tensor:
        other = self._ensure_tensor(other)
        return EQ.apply(self, other)

    # old
    # def all(self, dim: Optional[int] = None) -> Tensor:
    #     return All.apply(self, dim)
    # def all(self, dim: Optional[int] = None) -> Tensor:
    #     if dim is None:
    #         return All.apply(self)
    #     else:
    #         return All.apply(self, Tensor.make([dim], (1,), backend=self.backend))
    # new
    def all(self, dim: Optional[int] = None) -> Tensor:
        """Computes the logical AND over the entire tensor or along a specified dimension.

        Args:
        ----
            dim: The dimension along which to compute the logical AND. If None, computes over the entire tensor.

        Returns:
        -------
            Tensor: A tensor containing the logical AND of the input tensor.

        """
        if dim is None:
            return All.apply(self.view(self.size), self._ensure_tensor(0))
        else:
            return All.apply(self, self._ensure_tensor(dim))

    def is_close(self, other: Tensor) -> Tensor:
        """Computes the element-wise equality of two tensors within a tolerance.

        Args:
        ----
            other: The tensor to compare with.

        Returns:
        -------
            Tensor: A tensor containing the element-wise equality of the input tensors.

        """
        other = self._ensure_tensor(other)
        return IsClose.apply(self, other)

    def sigmoid(self) -> Tensor:
        """Applies the sigmoid function element-wise to the tensor.

        Returns
        -------
            Tensor: A tensor with the sigmoid function applied element-wise.

        """
        return Sigmoid.apply(self)

    def exp(self) -> Tensor:
        """Applies the exponential function element-wise to the tensor.

        Returns
        -------
            Tensor: A tensor with the exponential function applied element-wise.

        """
        return Exp.apply(self)

    def log(self) -> Tensor:
        """Applies the natural logarithm function element-wise to the tensor.

        Returns
        -------
            Tensor: A tensor with the natural logarithm function applied element-wise.

        """
        return Log.apply(self)

    def relu(self) -> Tensor:
        """Applies the ReLU (Rectified Linear Unit) activation function element-wise to the tensor.

        Returns
        -------
            Tensor: A tensor with the ReLU activation function applied element-wise.

        """
        return ReLU.apply(self)

    # old approach:
    # def sum(self, dim: Optional[int] = None) -> Tensor:
    #     if dim is None:
    #         # Sum over all dimensions
    #         return Sum.apply(self)
    #     else:
    #         # Wrap `dim` as a Tensor to satisfy `apply` method
    #         return Sum.apply(self, Tensor.make([dim], (1,), backend=self.backend))

    # New approach
    def sum(self, dim: Optional[int] = None) -> Tensor:
        """Compute the sum over dimension `dim`"""
        if dim is None:
            # reshaping the tensor to a 1D tensor before applying the reduction function:
            return Sum.apply(self.contiguous().view(self.size), self._ensure_tensor(0))
        else:
            return Sum.apply(self, self._ensure_tensor(dim))

    # old
    # def mean(self, dim: Optional[int] = None) -> Tensor:
    #     if dim is None:
    #         return Sum.apply(self) / self.size
    #     else:
    #         return Sum.apply(self, Tensor.make([dim], (1,), backend=self.backend)) / self.shape[dim]
    # new
    def mean(self, dim: Optional[int] = None) -> Tensor:
        """Compute the mean over dimension `dim`"""
        if dim is not None:
            return self.sum(dim) / self.shape[dim]
        else:
            return self.sum() / self.size

    # def permute(self, *dims: int) -> Tensor:
    #     """Permute the dimensions of the tensor.

    #     Args:
    #     ----
    #         *dims: The new order of dimensions.

    #     Returns:
    #     -------
    #         Tensor: A new tensor with permuted dimensions.

    #     """
    #     return Permute.apply(self, Tensor.make(list(dims), (len(dims),), backend=self.backend))
    def permute(self, *dims: int) -> Tensor:
        """Permute the dimensions of the tensor.

        Args:
        ----
            *dims: The new order of dimensions.

        Returns:
        -------
            Tensor: A new tensor with permuted dimensions.

        """
        return Permute.apply(self, tensor(list(dims)))

    # def view(self, *shape: int) -> Tensor:
    #     """Change the shape of the tensor to a new shape with the same size"""
    #     return View.apply(self, Tensor.make(list(shape), (len(shape),), backend=self.backend))

    def view(self, *shape: int) -> Tensor:
        """Change the shape of the tensor to a new shape with the same size"""
        return View.apply(self, tensor(list(shape)))
