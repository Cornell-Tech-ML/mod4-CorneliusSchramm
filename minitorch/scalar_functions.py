from __future__ import annotations

from typing import TYPE_CHECKING


import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Apply the scalar function to the given values.

        Args:
        ----
            vals: The values to apply the scalar function to.

        Returns:
        -------
            The result of the scalar function.

        """
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Addition function $f(x, y) = x + y$"""
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Computes the derivative of the addition function with respect to its inputs."""
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Logarithm function $f(x) = log(x)$"""
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the derivative of the logarithm function multiplied by a scalar"""
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


class Mul(ScalarFunction):
    """Multiplication function $f(x, y) = x * y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Multiplication function $f(x, y) = x * y$"""
        ctx.save_for_backward(a, b)
        return operators.mul(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Computes the derivative of the multiplication function with respect to its inputs."""
        a, b = ctx.saved_values
        return b * d_output, a * d_output


class Inv(ScalarFunction):
    """Inverse function $f(x) = 1/x"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Inverse function $f(x) = 1/x"""
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the derivative of the inverse function with respect to its input."""
        (a,) = ctx.saved_values
        return operators.inv_back(a, d_output)


class Neg(ScalarFunction):
    """Negative function $f(x) = -x"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Negative function $f(x) = -x"""
        return operators.neg(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the derivative of the negative function with respect to its input."""
        return (-1) * d_output


class Sigmoid(ScalarFunction):
    """Sigmoid function $f(x) = 1 / (1 + e^{-x})$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Sigmoid function $f(x) = 1 / (1 + e^{-x})$"""
        ctx.save_for_backward(a)
        return operators.sigmoid(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the derivative of the sigmoid function with respect to its input."""
        (a,) = ctx.saved_values
        s = operators.sigmoid(a)
        return s * (1.0 - s) * d_output


class ReLU(ScalarFunction):
    """ReLU function $f(x) = max(0, x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """ReLU function $f(x) = max(0, x)$"""
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the derivative of the ReLU function with respect to its input."""
        (a,) = ctx.saved_values
        return operators.relu_back(a, d_output)


class Exp(ScalarFunction):
    """Exponential function $f(x) = e^x"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Exponential function $f(x) = e^x"""
        ctx.save_for_backward(a)
        return operators.exp(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the derivative of the exponential function with respect to its input."""
        (a,) = ctx.saved_values
        return operators.exp(a) * d_output


class LT(ScalarFunction):
    """Less than function $f(x, y) = x < y"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Less than function $f(x, y) = x < y"""
        return operators.lt(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Computes the derivative of the less than function with respect to its inputs."""
        return (0.0, 0.0)


class GT(ScalarFunction):
    """Greater than function $f(x, y) = x > y"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Greater than function $f(x, y) = x > y"""
        return operators.gt(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Computes the derivative of the greater than function with respect to its inputs."""
        return (0.0, 0.0)


class EQ(ScalarFunction):
    """Equality function $f(x, y) = x == y"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Equality function $f(x, y) = x == y"""
        return operators.eq(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Computes the derivative of the equality function with respect to its inputs."""
        return (0.0, 0.0)


# TODO: Implement for Task 1.2.
