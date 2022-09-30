import numpy
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt


def check_array_length(array: []) -> bool:
    return len(array) == 3


def check_if_int(number: str) -> bool:
    try:
        int(number)
        return True
    except:
        return False


def is_line_correct(line: [str]) -> bool:
    return line and check_array_length(line) and check_if_int(line[0])


def generate_test_shit(k: float, b: float, c: int, n: int = 100) -> (np.array, np.array):
    return np.array([i * 1. for i in range(n)]), \
           np.array([b + k * i + np.random.randint(-c // 2, c // 2, 1) for i in range(n)])


def squares_sum(x: np.array, y: np.array, k: float, b: float):
    # res = .0
    # for i in range(x.size):
    #    res += (y[i] - k * x[i] - b) ** 2
    return np.sum((y - k * x - b) ** 2)


def square_sums_in_range(x: np.ndarray, y: np.ndarray, k: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array([[squares_sum(x, y, single_k, single_b) for single_k in k.flat] for single_b in b.flat])


def build_graph(x: np.array, y: np.array):
    plt.plot(x.tolist(), y.tolist(), '.r')
    plt.show()


def surface(matrix):
    plt.imshow(matrix)
    plt.gray()
    plt.show()
