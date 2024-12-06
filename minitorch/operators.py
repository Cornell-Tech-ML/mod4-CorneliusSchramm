"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, Iterable, List

# ## Task 0.1

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(x: float, y: float) -> float:
    """Multiplies two numbers.

    Args:
    ----
        x: The first number.
        y: The second number.

    Returns:
    -------
        float: The product of x and y.

    """
    return x * y


def id(x: float) -> float:
    """Returns the identity of a number.

    Args:
    ----
        x: The number to return.

    Returns:
    -------
        float: The number itself.

    """
    return x


def add(x: float, y: float) -> float:
    """Adds two numbers.

    Args:
    ----
        x: The first number.
        y: The second number.

    Returns:
    -------
        float: The sum of x and y.

    """
    return x + y


def neg(x: float) -> float:
    """Negates a number.

    Args:
    ----
        x: The number to negate.

    Returns:
    -------
        float: The negated number.

    """
    return -float(x)


def lt(x: float, y: float) -> float:
    """Compares two numbers chcking if one number is less than another

    Args:
    ----
        x: The first number.
        y: The second number.

    Returns:
    -------
        bool: True if x is less than y, False otherwise.

    """
    return 1.0 if x < y else 0.0


def gt(x: float, y: float) -> float:
    """Compares two numbers checking if one number is greater than another

    Args:
    ----
        x: The first number.
        y: The second number.

    Returns:
    -------
        bool: True if x is greater than y, False otherwise.

    """
    return 1.0 if x > y else 0.0


def eq(x: float, y: float) -> float:
    """Compares two numbers checking if they are equal.

    Args:
    ----
        x: The first number.
        y: The second number.

    Returns:
    -------
        bool: True if x is equal to y, False otherwise.

    """
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Returns the maximum of two numbers.

    Args:
    ----
        x: The first number.
        y: The second number.

    Returns:
    -------
        float: The maximum of x and y.

    """
    return float(x if x > y else y)


def is_close(x: float, y: float) -> float:
    """Checks if two numbers are close.

    Args:
    ----
        x: The first number.
        y: The second number.

    Returns:
    -------
        bool: True if x is close to y, False otherwise.

    """
    return 1.0 if abs(x - y) < 1e-2 else 0.0


def sigmoid(x: float) -> float:
    """Returns the sigmoid of a number.

    Args:
    ----
        x: The number to return the sigmoid of.

    Returns:
    -------
        float: The sigmoid of x.

    """
    return float(
        1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))
    )


def relu(x: float) -> float:
    """Returns the ReLU of a number.

    Args:
    ----
        x: The number to return the ReLU of.

    Returns:
    -------
        float: The ReLU of x.

    """
    return float(x if x >= 0 else 0.0)


def log(x: float) -> float:
    """Returns the natural logarithm of a number.

    Args:
    ----
        x: The number to return the natural logarithm of.

    Returns:
    -------
        float: The natural logarithm of x.

    """
    return float(math.log(x))


def exp(x: float) -> float:
    """Returns the exponential of a number.

    Args:
    ----
        x: The number to return the exponential of.

    Returns:
    -------
        float: The exponential of x.

    """
    return float(math.exp(x))


# def inv(x: float) -> float:
#     """Returns the inverse of a number.

#     Args:
#     ----
#         x: The number to return the inverse of.

#     Returns:
#     -------
#         float: The inverse of x.


#     """
#     return 1.0 / x if x != 0.0 else 0.0
def inv(x: float) -> float:
    """$f(x) = 1/x$"""
    return 1.0 / x


def inv_back(x: float, d: float) -> float:
    r"""If $f(x) = 1/x$ compute $d \times f'(x)$"""
    return (-d) / (x**2)


def log_back(x: float, d: float) -> float:
    """Computes the derivative of the natural logarithm function multiplied by a scalar.

    This function implements the chain rule for the derivative of ln(x),
    which is 1/x, and multiplies it by a scalar d.

    Args:
    ----
        x: The input value for which to compute the derivative of ln(x).
        d: A scalar value to multiply with the derivative.

    Returns:
    -------
        float: The result of (1/x) * d, which is the derivative of ln(x) times d.

    """
    return d / x


# def inv_back(x: float, d: float) -> float:
#     """Computes the derivative of the inverse function (1/x) multiplied by a scalar.

#     This function implements the chain rule for the derivative of 1/x,
#     which is -1/x^2, and multiplies it by a scalar d.

#     Args:
#     ----
#         x: The input value for which to compute the derivative of 1/x.
#         d: A scalar value to multiply with the derivative.

#     Returns:
#     -------
#         float: The result of (-1/x^2) * d, which is the derivative of 1/x times d.

#     """
#     return -(inv(x) ** 2) * d


def relu_back(x: float, d: float) -> float:
    """Computes the derivative of ReLU function multiplied by a scalar.

    This function calculates the derivative of the ReLU function
    at point x and multiplies it by a scalar d. The derivative of ReLU is 1 for x > 0
    and 0 for x <= 0.

    Args:
    ----
        x: The input value at which to compute the derivative of ReLU.
        d: A scalar value to multiply with the derivative.

    Returns:
    -------
        float: The result of the ReLU derivative at x multiplied by d.
               Returns d if x > 0, and 0 otherwise.

    """
    return d if x > 0.0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(
    function: Callable[[float], float],
) -> Callable[[Iterable[float]], Iterable[float]]:
    """Applies a given function to each element of a list.

    Args:
    ----
        function: The function to apply to each element.

    Returns:
    -------
        Callable: A function that takes a list and returns a new list with the function applied to each element.

    """

    def apply(lst: Iterable[float]) -> Iterable[float]:
        return [function(x) for x in lst]

    return apply


def zipWith(
    function: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Applies a function to corresponding elements of two lists.

    Args:
    ----
        function: The function to apply to each pair of elements.

    Returns:
    -------
        Callable: A function that takes two lists and returns a new list with the function applied to each pair of elements.

    """

    def apply(lst1: Iterable[float], lst2: Iterable[float]) -> Iterable[float]:
        return [function(x, y) for x, y in zip(lst1, lst2)]

    return apply


def reduce(
    function: Callable[[float, float], float], base_value: float
) -> Callable[[Iterable[float]], float]:
    """Applies a function to reduce a list to a single value.

    Args:
    ----
        function: The function to apply for reduction.
        base_value: The initial value for the reduction.

    Returns:
    -------
        Callable: A function that takes a list and returns the reduced value.

    """

    def apply(lst: Iterable[float]) -> float:
        accumulator = base_value
        for x in lst:
            accumulator = function(accumulator, x)
        return accumulator

    return apply


def negList(
    lst: List[float],
) -> List[float]:
    """Negates all elements in a list using map.

    Args:
    ----
        lst: The list to negate.

    Returns:
    -------
        list: A new list with all elements negated.

    """
    return list(map(neg)(lst))


def addLists(lst1: List[float], lst2: List[float]) -> List[float]:
    """Adds corresponding elements of two lists using zipWith.

    Args:
    ----
        lst1: The first list.
        lst2: The second list.

    Returns:
    -------
        list: A new list with corresponding elements added.

    """
    return list(zipWith(add)(lst1, lst2))


def sum(lst: List[float]) -> float:
    """Sums all elements in a list using reduce.

    Args:
    ----
        lst: The list to sum.

    Returns:
    -------
        float: The sum of the list.

    """
    if len(lst) == 0:
        return 0.0
    return reduce(add, 0.0)(lst)


def prod(lst: List[float]) -> float:
    """Multiplies all elements in a list using reduce.

    Args:
    ----
        lst: The list to multiply.

    Returns:
    -------
        float: The product of the list.

    """
    if len(lst) == 0:
        return 1.0
    return reduce(mul, 1.0)(lst)
