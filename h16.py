#! /usr/bin/env python3
"""Ridge polynomial regression."""
import torch
import math
from matplotlib import pyplot as plt

x = torch.arange(25) * 0.041
X = torch.vander(x, 7 + 1)
y_true = torch.sin(2 * math.pi * x)


def f(lamb: float) -> torch.Tensor:
    """f.

    :param lamb:
    :type lamb: float
    :rtype: torch.Tensor
    """
    y = y_true + 0.3 * torch.randn(25)

    lhs = X.T @ X
    rhs = X.T @ y
    ridge = lamb * torch.eye(lhs.shape[0])
    w = torch.linalg.lstsq(lhs + ridge, rhs).solution
    y_hat = X @ w
    return y_hat


for lamb in [0, 0.01, 0.1]:
    for _ in range(100):
        y_hat = f(lamb)
        plt.title(f"lambda = {lamb}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.plot(x, y_hat)
    plt.plot(x, y_true, "k", linewidth=3)
    plt.show()
