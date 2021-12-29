#! /usr/bin/env python3
"""Regression tree.
Implement an algorithm for building regression tree, where the leaf nodes are
3th-order polynomials rather than constants, use an adjustable threshold T to
early terminate tree building. Test your algorithm with one dataset that you
had generated in the Exercise 16. Observe the results with respect to T.

usage: h37.py [-hV] [-t <T>]

options:
    -h, --help              Show this screen.
    -V, --version           Show version.
    -t, --threshold <T>     Threshold of regession tree. [default: 0]

Threshold is the number of branchs. If it is zeros, the program will
demonstrate a possible thresholds.
"""
import torch
from torch.nn import functional as F
import math
from typing import Dict, Union, List
from torch import Tensor
from matplotlib import pyplot as plt
import os

xs = torch.arange(25) * 0.041
ys = torch.sin(2 * math.pi * xs) + 0.3 * torch.randn(25)


def f(x, y, lamb: float = 0):
    """f.
    Ridge polynomial regression. Referred from h16.py

    :param x:
    :param y:
    :param lamb:
    :type lamb: float
    """
    X = torch.vander(x, 3 + 1)
    lhs = X.T @ X
    rhs = X.T @ y
    ridge = lamb * torch.eye(lhs.shape[0])
    w = torch.linalg.lstsq(lhs + ridge, rhs).solution
    y_hat = X @ w
    loss = F.mse_loss(y_hat, y) * len(y)
    return w, loss


class DecisionTreeRegressor:
    """DecisionTreeRegressor."""

    def __init__(self, T: int):
        """__init__.

        :param T: the number of branchs
        :type T: int
        """
        self.T = T - 1

    def fit(self, xs, ys):
        """fit.

        :param xs: must be sorted
        :param ys:
        """
        split_indices = torch.zeros(self.T, dtype=torch.int)
        best_ws = [torch.zeros(4)] * (self.T + 1)
        for i in range(self.T):
            total_loss_min = float("inf")
            best_index = 0
            for index in range(len(xs)):
                if index in split_indices:
                    continue
                split_indices[i] = index
                current_split_indices = split_indices[: i + 1].sort().values
                remain = len(xs) - current_split_indices[-1]
                if remain <= 0:
                    break
                true_splits = torch.cat(
                    [
                        current_split_indices[0, None],
                        current_split_indices.diff(),
                        remain[None],
                    ]
                ).tolist()
                xs_splits = xs.split(true_splits)
                ys_splits = ys.split(true_splits)
                results = list(map(f, xs_splits, ys_splits))
                ws = list(map(lambda x: x[0], results))
                total_loss = sum(map(lambda x: x[1], results))
                if total_loss < total_loss_min:
                    total_loss_min = total_loss
                    best_ws = ws
                    best_index = index
            split_indices[i] = best_index
        # convert split_indices to split_thresholds
        self.split_thresholds = xs.index_select(0, split_indices)
        self.best_ws = best_ws

    def predict(self, xs):
        return list(map(self.predict_one, xs))

    def predict_one(self, x):
        piecewise_index = (self.split_thresholds < x).sum()
        ws = self.best_ws[piecewise_index]
        X = torch.vander(x[None], len(ws)).squeeze()
        return ws.dot(X)


def draw(T: int):
    regressor = DecisionTreeRegressor(T)
    regressor.fit(xs, ys)
    x_grid = torch.arange(xs.min().item(), xs.max().item(), 0.001)
    y_grid = regressor.predict(x_grid)
    name = "decision_tree_regressor"
    fig, ax = plt.subplots(num=name)
    ax.scatter(xs, ys, color="red", label="$y$")
    ax.plot(x_grid, torch.sin(2 * math.pi * x_grid), label="$\grave{y}$")
    ax.plot(x_grid, y_grid, label="$\hat{y}$")
    ax.vlines(
        regressor.split_thresholds, -3, 3, colors="green", linestyles="dashed"
    )
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    plt.legend()
    path = os.path.join("images", "37")
    try:
        os.makedirs(path, exist_ok=True)
    except FileExistsError:
        pass
    plt.savefig(os.path.join(path, f"{regressor.T + 1}.png"))
    plt.show()

if __name__ == "__main__" and __doc__:
    from docopt import docopt

    Arg = Union[bool, str]
    args: Dict[str, Arg] = docopt(__doc__, version="v0.0.1")

    T = int(args["--threshold"])
    if T > 0:
        draw(T)
    else:
        for T in [3, 4, 5]:
            draw(T)
