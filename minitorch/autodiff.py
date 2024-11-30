from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(
    f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6
) -> float:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    vals_plus = list(vals)
    vals_plus[arg] += epsilon
    vals_minus = list(vals)
    vals_minus[arg] -= epsilon
    return (f(*vals_plus) - f(*vals_minus)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Interface method: Accumulates the derivative of this variable.

        Args:
        ----
            x: The derivative to accumulate.

        """
        ...

    @property
    def unique_id(self) -> int:
        """Unique identifier for this variable.

        Returns
        -------
            The unique identifier.

        """
        ...  # type: ignore

    def is_leaf(self) -> bool:
        """Interface method: Checks if this variable is a leaf.

        Returns
        -------
            True if this variable is a leaf, False otherwise.

        """
        ...

    def is_constant(self) -> bool:
        """Interface method: Checks if this variable is a constant.

        Returns
        -------
            True if this variable is a constant, False otherwise.

        """
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Interface method: Returns the parent variables of this variable.

        Returns
        -------
            The parent variables of this variable.

        """
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Interface method: Applies the chain rule to the derivative of this variable.

        Args:
        ----
            d_output: The derivative of the output with respect to this variable.

        Returns:
        -------
            The derivatives of the input variables with respect to this variable.

        """
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    result = []
    visited = {}  # Use a dict instead of a set

    def depth_first_search(var: Variable) -> None:
        if visited.get(var.unique_id, False):
            return
        visited[var.unique_id] = True
        if not var.is_constant():
            for parent in var.parents:
                depth_first_search(parent)
            result.append(var)

    depth_first_search(variable)
    return reversed(result)

    # raise NotImplementedError("Need to implement for Task 1.4")


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
    ----
        variable: The right-most variable
        deriv: The derivative of the final output with respect to the last variable

    Returns:
    -------
        None

    Note:
    ----
        This function writes its results to the derivative values of each leaf
        through `accumulate_derivative`.

    """
    order = topological_sort(variable)
    grad_map = {variable: deriv}

    for var in order:
        if var in grad_map:
            current_grad = grad_map[var]
            if not var.is_leaf():
                local_grads = var.chain_rule(current_grad)
                for input_var, grad in local_grads:
                    if input_var in grad_map:
                        grad_map[input_var] += grad
                    else:
                        grad_map[input_var] = grad
            else:
                var.accumulate_derivative(current_grad)


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Get the saved tensors from the forward pass.

        Returns
        -------
            The saved tensors.

        """
        return self.saved_values
