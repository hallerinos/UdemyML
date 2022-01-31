from typing import Any, Dict, Iterable, Tuple, Union, get_args
from matplotlib.collections import PathCollection
import numpy as np
import matplotlib.pyplot as plt


def generate_data(nscs: list = [13, 9, 6], **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
    means = kwargs.pop('means', [[0, 0], [10, 10], [0, 20]])
    cov = kwargs.pop('cov', [[1, 0], [0, 1]])
    tags = [i for i in range(len(nscs))]
    data = []
    for (nsc, mean) in zip(nscs, means):
        d = np.asarray(np.random.multivariate_normal(mean, cov, size=nsc))
        data.append(d)
    data = np.concatenate(data, axis=0)
    class_ids = np.concatenate([nsc * [tag] for (nsc, tag) in zip(nscs, tags)])
    return np.asarray(data), np.asarray(class_ids)


def plot_data(data: np.ndarray, class_ids: np.ndarray, **kwargs: Any) -> PathCollection:
    colors = kwargs.pop('colors', ["r", "b", "g", "m"])
    size = kwargs.pop('size', 100)
    marker = kwargs.pop('marker', 'o')
    plt.scatter(data[:, 0], data[:, 1], color=[colors[tag] for tag in class_ids], s=size, marker=marker)


def main():
    data, class_ids = generate_data()
    plot_data(data, class_ids)


if __name__ == "__main__":
    main()
    plt.show()
