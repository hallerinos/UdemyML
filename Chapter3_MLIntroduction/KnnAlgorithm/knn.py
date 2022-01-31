from typing import Any
import matplotlib.pyplot as plt
from dataset import generate_data, plot_data
import numpy as np


class KNNClassifier:
    def __init__(self, nns: int = 5) -> None:
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
        if len(y1) != len(y2):
            print("y1 and y2 have incompatible shapes.")
            return
        hits = [0]*len(y1)
        for (idx, (y1, y2)) in enumerate(zip(y1, y2)):
            if y1 == y2: hits[idx] = 1
        return 100 * np.mean(hits)


def main():
    plt.figure()
    x_train, y_train = generate_data([5, 5, 5], variance=[[10, 0], [0, 10]])
    plot_data(x_train, y_train, size=100, marker='h', colors=['black' for i in range(3)])
    plot_data(x_train, y_train, size=60, marker='h')

    knn = KNNClassifier(3)
    knn.fit(x_train, y_train)

    x, y = generate_data([10, 10, 10], variance=[[20, 0], [0, 10]])
    y_pred = knn.predict(x)
    plot_data(x, y, size=100, marker='o')
    plot_data(x, y_pred, size=25, marker='o')
    plt.title(f'Prediction accuracy: {np.around(knn.score(y_pred, y), 2)}%')
    plt.show()


if __name__ == "__main__":
    main()
