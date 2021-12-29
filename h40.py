#! /usr/bin/env python3
"""Homework 40.
Implement the naive Bayes algorithm (with Laplace smoothing and adjustable
:math:`\alpha`) by yourself. Use the dataset in Table 3 for training,
and :math:`x = (1, M)` for test. Give the results when :math:`alpha = 0` and
:math:`alpha = 1`.

Table 3: Dataset for binary classification with discrete input

x1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3
x2, S, M, M, S, S, S, M, M, L, L, L
y, -1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1

usage: h40.py [-hV] [-x <x1>] [-y <x2>] -a <alpha>

options:
    -h, --help                  Show this screen.
    -V, --version               Show version.
    -a, --alpha <alpha>         Adjustable alpha for Laplace smoothing.
    -x, --x1 <x1>               x1. [default: 1]
    -y, --x2 <x2>               x2. [default: m]


$ ./h40.py -a 1
{1: 0.059829059829059825, -1: 0.08653846153846154}
$ ./h40.py -a 0
{1: 0.0606060606060606, -1: 0.10909090909090909}
"""

if __name__ == "__main__" and __doc__:
    from docopt import docopt
    from typing import Dict

    try:
        args: Dict[str, str] = docopt(__doc__, version="v0.0.1")
    except Exception:
        args = {}

    # https://github.com/Rhythmblue/statistical_learning_homework/blob/master/exercise/Ex_6_1.py
    train_data = [
        [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3],
        ["s", "m", "m", "s", "s", "s", "m", "m", "l", "l", "l"],
    ]
    train_label = [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1]
    test_data = [int(args.get("--x1", 1)), args.get("--x2", 'm').lower()]
    alpha = float(args.get("--alpha", 1))

    sample_num = len(train_label)
    label_class = len(set(train_label))
    prob = {}
    for label in list(set(train_label)):
        prob_now = 1
        down = 0
        for i, element in enumerate(test_data):
            up = 0
            down = 0
            feature_class = len(set(train_data[i]))
            for j in range(sample_num):
                if train_label[j] == label:
                    down += 1
                    if train_data[i][j] == element:
                        up += 1
            prob_now *= (up + alpha) / (down + feature_class * alpha)
        prob_now *= (down + alpha) / (sample_num + label_class * alpha)
        prob[label] = prob_now

    print(prob)
