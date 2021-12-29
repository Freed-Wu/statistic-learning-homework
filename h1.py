#! /usr/bin/env python3
"""
$ ./main
[
    (-0.1750451624393463, tensor(0.7368)),
    (0.1280250996351242, tensor(1.0994)),
    (0.0201797503978014, tensor(0.9745)),
    (0.00458096107468009, tensor(1.0019)),
    (0.00020308303646743298, tensor(0.9988))
]
"""
import torch
def f(N=10, mu=0, sigma=1):
    X = mu + sigma * torch.randn(N)
    return X.mean().item(), X.std()

print(list(map(f, [10, 100, 1000, 10000, 100000])))
