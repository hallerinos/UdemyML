from typing import Iterable, Tuple, Union
from matplotlib.collections import PathCollection
import numpy as np
import matplotlib.pyplot as plt


def generate_data(num_points_per_class: list[int] = [13, 9, 6], means: list[np.ScalarType] = [[0, 0], [10, 10], [0, 20]], variance: list[list[np.ScalarType]] = [[1, 0], [0, 1]]) -> Tuple[np.ndarray, np.ndarray]:
    nscs = num_points_per_class
    tags = [i for i in range(len(nscs))]
    means = means
    cov = variance
    data = []
    for (nsc, mean) in zip(nscs, means):
        d = np.asarray(np.random.multivariate_normal(mean, cov, size=nsc))
        data.append(d)
    data = np.concatenate(data, axis=0)
    class_ids = np.concatenate([nsc * [tag] for (nsc, tag) in zip(nscs, tags)])
    return np.asarray(data), np.asarray(class_ids)


def plot_data(data: np.ndarray, class_ids: np.ndarray, colors: list[str] = ["r", "b", "g", "m"], size: np.ScalarType = 100, marker: str = 'o') -> PathCollection:
    plt.scatter(data[:, 0], data[:, 1], color=[colors[tag] for tag in class_ids], s=size, marker=marker)


def main():
    data, class_ids = generate_data()
    plot_data(data, class_ids)


if __name__=="__main__":
    main()
    plt.show()
