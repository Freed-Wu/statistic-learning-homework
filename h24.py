#! /usr/bin/env python3
"""Perceptron.

Implement the perceptron learning algorithm by yourself. Use the dataset in
Table 2 to test your algorithm.

Table 2: Dataset for binary classification on 2-D plane
xi (1, 2) (2, 3) (3, 3) (2, 1) (3, 2)
yi 1 1 1 -1 -1
"""
import torch
from torch import nn
from matplotlib import pyplot as plt


class Perceptron(nn.Module):
    """Perceptron."""

    def __init__(self, x, y, lr=1):
        """__init__.

        :param x:
        :param y:
        :param lr:
        """
        super().__init__()
        self.x = x
        self.y = y
        self.w = torch.zeros(x.shape[1])
        self.b = 0
        self.lr = lr

    def forward(self, x):
        """forward.

        :param x:
        """
        return torch.sign(torch.dot(x, self.w) + self.b)

    def training_step(self):
        """training_step."""
        for _ in range(10):
            for i in range(len(self.x)):
                y_hat = self(x[i, :])
                if y_hat * self.y[i] <= 0:
                    self.w += self.y[i] * self.lr * self.x[i, :]
                    self.b += self.y[i]
        return self.w, self.b


x = torch.tensor([[1, 2], [2, 3], [3, 3], [2, 1], [3, 2]]) + 0.0
y = torch.tensor([1, 1, 1, -1, -1])
w, b = Perceptron(x, y).training_step()

plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
x1s = torch.arange(0, 5, 0.1)
x2s = (-b - w[0] * x1s) / w[1]
plt.plot(x1s, x2s)
cls1 = x[y == 1, :]
cls0 = x[y == -1, :]
plt.scatter(cls1[:, 0], cls1[:, 1], c="r")
plt.scatter(cls0[:, 0], cls0[:, 1], c="b")
plt.show()
