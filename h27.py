#! /usr/bin/env python3
"""SVM.

Table 2: Dataset for binary classification on 2-D plane
xi (1, 2) (2, 3) (3, 3) (2, 1) (3, 2)
yi 1 1 1 −1 −1
"""
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import SGD
from matplotlib import pyplot as plt
import matplotlib as mpl

X_train = np.array([[1, 2], [2, 3], [3, 3], [2, 1], [3, 2]])
y_train = np.array([1, 1, 1, -1, -1])


class SVM(nn.Module):
    """SVM."""

    def __init__(self):
        """__init__."""
        super().__init__()
        self.layer = nn.Linear(2, 1)

    def forward(self, x):
        """forward.

        :param x:
        """
        x = self.layer(x)
        return x


def loss_func(scores, label):
    """loss_func.

    :param scores:
    :param label:
    """
    loss = 1 - label * scores
    loss[loss <= 0] = 0
    return torch.sum(loss)


def sign(x):
    """sign.

    :param x:
    """
    x[x >= 0] = 1
    x[x < 0] = -1
    return x


def pred(x):
    """pred.

    :param x:
    """
    return sign(x)


model = SVM()

optim_func = SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    inputs, targets = X_train, y_train
    inputs = Variable(torch.from_numpy(inputs).float(), requires_grad=False)
    label = Variable(torch.from_numpy(targets).int(), requires_grad=False)
    scores = model(inputs).squeeze(1)
    loss = loss_func(scores, label)
    optim_func.zero_grad()
    loss.backward()
    optim_func.step()


if __name__ == "__main__":
    colors = ["red", "blue"]
    w1, w2 = model.state_dict()["layer.weight"][0]
    b = model.state_dict()["layer.bias"][0]
    print(w1, w2, b)
    x1 = np.linspace(1, 3, 100)
    x2 = (-b - w1 * x1) / w2
    fig, ax = plt.subplots()
    ax.plot(x1, x2)
    markers = ("x", "o")
    for idx, ci in enumerate(np.unique(y_train)):
        ax.scatter(
            x=X_train[y_train == ci, 0],
            y=X_train[y_train == ci, 1],
            alpha=0.8,
            marker=markers[idx],
            label=ci,
        )

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    plt.legend()
    plt.savefig("svm.png")
    plt.show()
