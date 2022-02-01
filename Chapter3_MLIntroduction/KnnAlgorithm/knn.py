from typing import Any
import matplotlib.pyplot as plt
from dataset import generate_data, plot_data
import numpy as np


class KNNClassifier:
    def __init__(self, nns: int = 3) -> None:
        self.nns = nns
        self.x : np.ndarray
        self.y : np.ndarray
        self.num_classes : int

    def _distance(self, p1: np.ndarray, p2: np.ndarray) -> Any:
        return np.linalg.norm(p1 - p2)

    def kneibors(self, x: np.ndarray) -> np.ndarray:
        dists = np.array([[self._distance(xi, xj) for xi in self.x] for xj in x])
        nn_dists = np.array([np.argsort(d)[0:self.nns] for d in dists])
        return nn_dists

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = x
        self.y = y
        self.num_classes = len(np.unique(self.y))

    def predict(self, x: np.ndarray) -> np.ndarray:
        n_idxs = self.kneibors(x)
        y_nns = self.y[n_idxs]
        y_pred = np.array([int(np.round(np.mean(y))) for y in y_nns])
        return y_pred

    def score(self, y1: np.ndarray, y2: np.ndarray) -> np.ScalarType:
        hits = (y1 == y2)
        return np.mean(hits)

    def optimize_nns(self, p: np.ScalarType = 0.8) -> None:
        x, y = self.x, self.y
        ridxs = np.random.choice(len(x), int(p * len(x)))

        # fit only a fraction p of the total data, rest is used to optimize nns
        x_slice, y_slice = x[ridxs], y[ridxs]
        self.fit(x_slice, y_slice)

        x_conj, y_conj = np.delete(x, ridxs, axis=0), np.delete(y, ridxs)
        scores = []
        for nn in range(3, 10):
            self.nns = nn
            y_pred = self.predict(x_conj)
            score = self.score(y_pred, y_conj)
            scores.append(score)
            if abs(score - 1.0 / self.num_classes) < 0.1:
                break
        print(scores)
        best_idx = np.flip(np.argsort(scores))[0]
        self.nns = best_idx + 3  # set nn value with best score
        self.fit(x, y)  # fit the full data again


def main():
    # create training data
    x_train, y_train = generate_data([20, 20, 20], cov=[[20, 0], [0, 20]])

    # create knn object and optimize nns
    knn = KNNClassifier()
    knn.fit(x_train, y_train)
    knn.optimize_nns()
    print(f"Optimized nns value: {knn.nns}")

    # use optimized model for prediction
    x, y = generate_data([20, 20, 20], cov=[[20, 0], [0, 20]])
    y_pred = knn.predict(x)
    score = knn.score(y_pred, y)

    # prepare combined figure
    plt.figure()
    plot_data(x_train, y_train, size=150, marker='h', colors=['black' for i in range(3)])
    plot_data(x_train, y_train, size=60, marker='h')
    plot_data(x, y, size=100, marker='o')
    plot_data(x, y_pred, size=20, marker='o')
    plt.title(f'Prediction accuracy: {np.around(score, 3)}')
    plt.show()


if __name__ == "__main__":
    main()
