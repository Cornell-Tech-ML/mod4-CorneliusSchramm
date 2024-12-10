import math
import random
from dataclasses import dataclass
from typing import List, Tuple


def make_pts(N: int) -> List[Tuple[float, float]]:
    """Generate N random 2D points with coordinates between 0 and 1.

    Args:
    ----
        N: Number of points to generate.

    Returns:
    -------
        List: A list of N 2D points with coordinates between 0 and 1.

    """
    X = []
    for i in range(N):
        x_1 = random.random()
        x_2 = random.random()
        X.append((x_1, x_2))
    return X


@dataclass
class Graph:
    """A graph of N points that can be used for classification tasks.

    This class represents a dataset consisting of N points in a 2D space,
    along with their corresponding labels.

    Attributes
    ----------
        N: The number of points in the graph.
        X: A list of 2D points represented as tuples.
        y: A list of labels corresponding to each point in X.

    """

    N: int
    X: List[Tuple[float, float]]
    y: List[int]


def simple(N: int) -> Graph:
    """Generate a simple linear dataset of N points that can be classified by a line.

    Args:
    ----
        N: Number of points to generate.

    Returns:
    -------
        Graph: A graph containing N, the points (X) and their corresponding labels (y).

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def diag(N: int) -> Graph:
    """Generate a diagonal dataset of N points classified based on their position relative to the line y = -x + 0.5.

    This function creates a dataset where each point is classified as 1 if it lies below the line
    defined by the equation x1 + x2 < 0.5, and as 0 if it lies on or above that line. The slope of the
    line is -1, creating a diagonal separation in the dataset.

    Args:
    ----
        N (int): Number of points to generate.

    Returns:
    -------
        Graph: A graph containing N, the points (X) and their corresponding labels (y).

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 + x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def split(N: int) -> Graph:
    """Generate a split dataset of N points.

    This function creates a dataset where points are classified into two categories
    based on their x-coordinate. Points with x < 0.2 or x > 0.8 are labeled as 1,
    and points with 0.2 <= x <= 0.8 are labeled as 0. This creates a 'split' effect
    in the dataset, with two separate regions for class 1.

    Args:
    ----
        N: Number of points to generate.

    Returns:
    -------
        Graph: A graph containing N, the points (X) and their corresponding labels (y).

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.2 or x_1 > 0.8 else 0
        y.append(y1)
    return Graph(N, X, y)


def xor(N: int) -> Graph:
    """Generate an XOR dataset of N points.

    This function creates a dataset where points are classified into two categories
    based on the XOR-like relationship between their x and y coordinates. Specifically, points
    where one coordinate is less than 0.5 and the other is greater than 0.5 are labeled as 1.
    All other points are labeled as 0.

    Args:
    ----
        N: Number of points to generate.

    Returns:
    -------
        Graph: A graph containing N, the points (X) and their corresponding labels (y).

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 and x_2 > 0.5 or x_1 > 0.5 and x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def circle(N: int) -> Graph:
    """Generate a circle dataset of N points.

    This function creates a dataset where points are classified into two categories
    based on their distance from the center of a circle. Points inside the circle
    are labeled as 1, and points outside the circle are labeled as 0.

    Args:
    ----
        N: Number of points to generate.

    Returns:
    -------
        Graph: A graph containing N, the points (X) and their corresponding labels (y).

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        x1, x2 = x_1 - 0.5, x_2 - 0.5
        y1 = 1 if x1 * x1 + x2 * x2 > 0.1 else 0
        y.append(y1)
    return Graph(N, X, y)


def spiral(N: int) -> Graph:
    """Generate a spiral dataset of N points.

    This function creates a spiral dataset where the points are classified into two
    categories based on their position along two intertwined spirals. Each class forms
    a distinct spiral shape. One spiral corresponds to points labeled as 0, while the other
    spiral consists of points labeled as 1.

    Args:
    ----
        N: Number of points to generate.

    Returns:
    -------
        Graph: A graph containing N, the points (X) and their corresponding labels (y).

    """

    def x(t: float) -> float:
        """Calculate the scaled x-coordinate of a point in a spiral.

        Args:
        ----
            t: The parameter that controls both the angle and the distance from the center
           (essentially defining the spiral).

        Returns:
        -------
            float: The x-coordinate of the point after applying a scaling factor to control the spiral's size.

        """
        return t * math.cos(t) / 20.0

    def y(t: float) -> float:
        """Calculate the scaled y-coordinate of a point in a spiral.

        Args:
        ----
        t: The parameter that controls both the angle and the distance from the center
           (essentially defining the spiral).

        Returns:
        -------
            float: The y-coordinate of the point after applying a scaling factor to control the spiral's size.

        """
        return t * math.sin(t) / 20.0

    X = [
        (x(10.0 * (float(i) / (N // 2))) + 0.5, y(10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    X = X + [
        (y(-10.0 * (float(i) / (N // 2))) + 0.5, x(-10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    y2 = [0] * (N // 2) + [1] * (N // 2)
    return Graph(N, X, y2)


datasets = {
    "Simple": simple,
    "Diag": diag,
    "Split": split,
    "Xor": xor,
    "Circle": circle,
    "Spiral": spiral,
}
