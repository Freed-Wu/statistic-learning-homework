#! /usr/bin/env python3
"""36.

Implement the AdaBoost algorithm, with decision stamp as base learner, by
yourself. Use the dataset in Table 3 for training, and x = (1, M ) for test.
"""
import numpy as np

class AdaBoost:
    """AdaBoost."""

    def __init__(self, n_estimators=50, learning_rate=1.0):
        """__init__.

        :param n_estimators:
        :param learning_rate:
        """
        self.clf_num = n_estimators
        self.learning_rate = learning_rate

    def _G(self, features, labels, weights):
        """_G.

        :param features:
        :param labels:
        :param weights:
        """
        m = len(features)
        error = float('inf')
        best_v = 0.0
        features_min = min(features)
        features_max = max(features)
        n_step = (features_max - features_min +
                  self.learning_rate) // self.learning_rate
        direct, compare_array = None, None
        for i in range(1, int(n_step)):
            v = features_min + self.learning_rate * i

            if v not in features:
                compare_array_positive = np.array(
                    [1 if features[k] > v else -1 for k in range(m)])
                weight_error_positive = sum([
                    weights[k] for k in range(m)
                    if compare_array_positive[k] != labels[k]
                ])

                compare_array_nagetive = np.array(
                    [-1 if features[k] > v else 1 for k in range(m)])
                weight_error_nagetive = sum([
                    weights[k] for k in range(m)
                    if compare_array_nagetive[k] != labels[k]
                ])

                if weight_error_positive < weight_error_nagetive:
                    weight_error = weight_error_positive
                    _compare_array = compare_array_positive
                    direct = 'positive'
                else:
                    weight_error = weight_error_nagetive
                    _compare_array = compare_array_nagetive
                    direct = 'nagetive'

                if weight_error < error:
                    error = weight_error
                    compare_array = _compare_array
                    best_v = v
        return best_v, direct, error, compare_array

    def _alpha(self, error):
        """_alpha.

        :param error:
        """
        return 0.5 * np.log((1 - error) / error)

    def _Z(self, weights, a, clf):
        """_Z.

        :param weights:
        :param a:
        :param clf:
        """
        return sum([
            weights[i] * np.exp(-1 * a * self.Y[i] * clf[i])
            for i in range(self.M)
        ])

    def _w(self, a, clf, Z):
        """_w.

        :param a:
        :param clf:
        :param Z:
        """
        for i in range(self.M):
            self.weights[i] = self.weights[i] * np.exp(
                -1 * a * self.Y[i] * clf[i]) / Z

    def G(self, x, v, direct):
        """G.

        :param x:
        :param v:
        :param direct:
        """
        if direct == 'positive':
            return 1 if x > v else -1
        else:
            return -1 if x > v else 1

    def fit(self, X, y):
        """fit.

        :param X:
        :param y:
        """
        self.X = X
        self.Y = y
        self.M, self.N = X.shape
        self.clf_sets = []
        self.weights = [1.0 / self.M] * self.M
        self.alpha = []

        for _ in range(self.clf_num):
            best_clf_error, best_v, clf_result = 100000, None, None
            for j in range(self.N):
                features = self.X[:, j]
                v, direct, error, compare_array = self._G(
                    features, self.Y, self.weights)

                if error < best_clf_error:
                    best_clf_error = error
                    best_v = v
                    final_direct = direct
                    clf_result = compare_array
                    axis = j

                if best_clf_error == 0:
                    break

            a = self._alpha(best_clf_error)
            self.alpha.append(a)
            self.clf_sets.append((axis, best_v, final_direct))
            Z = self._Z(self.weights, a, clf_result)
            self._w(a, clf_result, Z)

    def predict(self, feature):
        """predict.

        :param feature:
        """
        result = 0.0
        for i in range(len(self.clf_sets)):
            axis, clf_v, direct = self.clf_sets[i]
            f_input = feature[axis]
            result += self.alpha[i] * self.G(f_input, clf_v, direct)
        return 1 if result > 0 else -1


S, M, L = 1, 2, 3
X = np.array([[1, S], [1, M], [1, M], [1, S], [1, S], [2, S], [2, M], [2, M], [2, L], [2, L], [3, L]])
y = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1])

clf = AdaBoost(n_estimators=3, learning_rate=0.5)
clf.fit(X, y)
print(clf.predict([1, M]))
