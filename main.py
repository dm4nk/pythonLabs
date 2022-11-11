import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from numpy.linalg import inv


def test_data(k: float = 1.0, b: float = 0.1, rand_range: float = 10.0, n: int = 100) -> (np.array, np.array):
    return np.array([i / n for i in range(n)]), \
           np.array([b + k * i ** 2 + random.uniform(-rand_range * 0.5, rand_range * 0.5) for i in range(n)])


def test_data_2d(kx: float = -2.0, ky: float = 2.0, b: float = 12.0, rand_range: float = 100.0, n: int = 100) -> \
        (np.ndarray, np.ndarray, np.ndarray):
    """
    Генерирует плоскость вида z = kx*x + ky*x + b + dz, где dz - аддитивный шум с амплитудой half_disp
    :param kx: наклон плоскости по x
    :param ky: наклон плоскости по y
    :param b: смещение по z
    :param rand_range: амплитуда разброса данных
    :param n: количество точек
    :returns: кортеж значенией по x, y и z
    """
    x = np.array([random.uniform(0.0, n * 1.) for i in range(n)])
    y = np.array([random.uniform(0.0, n * 1.) for i in range(n)])
    dz = np.array([b + random.uniform(-rand_range * 0.5, rand_range * 0.5) for i in range(n)])
    return x, y, x * kx + y * ky + dz


def test_data_quad(surf_params=np.array([1, 2, 3, 1, 2, 3]), n: int = 100, rand_range: float = 100.0, b: float = 12.0):
    x = np.array([random.uniform(0.0, n * 1.) for i in range(n)])
    y = np.array([random.uniform(0.0, n * 1.) for i in range(n)])
    dz = np.array([b + random.uniform(-rand_range * 0.5, rand_range * 0.5) for i in range(n)])

    F = surf_params[0] * x ** 2 + surf_params[1] * y ** 2 + surf_params[2] * x * y + surf_params[3] * x + surf_params[
        4] * y + surf_params[5]

    return x, y, dz + F


def distance_sum(x: np.ndarray, y: np.ndarray, k: float, b: float) -> float:
    """
    Вычисляет сумму квадратов расстояний от набора точек до линии вида y = k*x + b при фиксированных k и b
    по формуле: F(k, b) = (Σ(yi -(k * xi + b))^2)^0.5 (суммирование по i)
    :param x: массив значений по x
    :param y: массив значений по y
    :param k: значение параметра k (наклон)
    :param b: значение параметра b (смещение)
    :returns: F(k, b) = (Σ(yi -(k * xi + b))^2)^0.5
    """
    return np.sqrt(np.sum((y - x * k + b) ** 2))


def distance_field(x: np.ndarray, y: np.ndarray, k: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Вычисляет сумму квадратов расстояний от набора точек до линии вида y = k*x + b, где k и b являются диапазонами
    значений. Формула расстояния для j-ого значения из набора k и k-ого значения из набора b:
    F(k_j, b_k) = (Σ(yi -(k_j * xi + b_k))^2)^0.5 (суммирование по i)
    :param x: массив значений по x
    :param y: массив значений по y
    :param k: массив значений параметра k (наклоны)
    :param b: массив значений параметра b (смещения)
    :returns: поле расстояний вида F(k, b) = (Σ(yi -(k * xi + b))^2)^0.5 (суммирование по i)
    """
    return np.array([[distance_sum(x, y, k_i, b_i) for k_i in k.flat] for b_i in b.flat])


def linear_regression(x: np.ndarray, y: np.ndarray) -> [float, float]:
    """
    Линейная регрессия.\n
    Основные формулы:\n
    yi - xi*k - b = ei\n
    yi - (xi*k + b) = ei\n
    (yi - (xi*k + b))^2 = yi^2 - 2*yi*(xi*k + b) + (xi*k + b)^2 = ei^2\n
    yi^2 - 2*(yi*xi*k + yi*b) + (xi^2 * k^2 + 2 * xi * k * b + b^2) = ei^2\n
    yi^2 - 2*yi*xi*k - 2*yi*b + xi^2 * k^2 + 2 * xi * k * b + b^2 = ei^2\n
    d ei^2 /dk = - 2*yi*xi + 2 * xi^2 * k + 2 * xi * b = 0\n
    d ei^2 /db = - 2*yi + 2 * xi * k + 2 * b = 0\n
    ====================================================================================================================\n
    d ei^2 /dk = (yi - xi * k - b) * xi = 0\n
    d ei^2 /db =  yi - xi * k - b = 0\n
    ====================================================================================================================\n
    Σ(yi - xi * k - b) * xi = 0\n
    Σ yi - xi * k - b = 0\n
    ====================================================================================================================\n
    Σ(yi - xi * k - b) * xi = 0\n
    Σ(yi - xi * k) = n * b\n
    ====================================================================================================================\n
    Σyi - k * Σxi = n * b\n
    Σxi*yi - xi^2 * k - xi*b = 0\n
    Σxi*yi - Σxi^2 * k - Σxi*b = 0\n
    Σxi*yi - Σxi^2 * k - Σxi*(Σyi - k * Σxi) / n = 0\n
    Σxi*yi - Σxi^2 * k - Σxi*Σyi / n + k * (Σxi)^2 / n = 0\n
    Σxi*yi - Σxi*Σyi / n + k * ((Σxi)^2 / n - Σxi^2)  = 0\n
    Σxi*yi - Σxi*Σyi / n = -k * ((Σxi)^2 / n - Σxi^2)\n
    (Σxi*yi - Σxi*Σyi / n) / (Σxi^2 - (Σxi)^2 / n) = k\n
    окончательно:\n
    k = (Σxi*yi - Σxi*Σyi / n) / (Σxi^2 - (Σxi)^2 / n)\n
    b = (Σyi - k * Σxi) /n\n
    :param x: массив значений по x
    :param y: массив значений по y
    :returns: возвращает пару (k, b), которая является решением задачи (Σ(yi -(k * xi + b))^2)->min
    """
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_xx = np.sum(x ** 2)
    n = len(x)
    k = (sum_xy - sum_x * sum_y / n) / (sum_xx - sum_x * sum_x / n)
    b = (sum_y - k * sum_x) / n
    return k, b


def bi_linear_regression(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> [float, float, float]:
    """
    Билинейная регрессия.\n
    Основные формулы:\n
    zi - (yi * ky + xi * kx + b) = ei\n
    zi^2 - 2*zi*(yi * ky + xi * kx + b) + (yi * ky + xi * kx + b)^2 = ei^2\n
    ei^2 = zi^2 - 2*yi*zi*ky - 2*zi*xi*kx - 2*zi*b + ((yi*ky)^2 + 2 * (xi*kx*yi*ky + b*yi*ky) + (xi*kx + b)^2)\n
    ei^2 = zi^2 - 2*yi*zi*ky - 2*zi*xi*kx - 2*zi*b + (yi*ky)^2 + 2*xi*kx*yi*ky + 2*b*yi*ky + (xi*kx + b)^2\n
    ei^2 =\n
    zi^2 - 2*zi*yi*ky - 2*zi*xi*kx - 2*zi*b + (yi*ky)^2 + 2*xi*kx*yi*ky + 2*b*yi*ky + (xi*kx)^2 + 2*xi*kx*b+ b^2\n
    ei^2 =\n
    zi^2 - 2*zi*yi*ky - 2*zi*xi*kx - 2*zi*b + (yi*ky)^2 + 2*xi*kx*yi*ky + 2*b*yi*ky + (xi*kx)^2 + 2*xi*kx*b+ b^2\n
    ei^2 =\n
    zi^2 - 2*zi*yi*ky - 2*zi*xi*kx - 2*zi*b + (yi*ky)^2 + 2*xi*kx*yi*ky + 2*b*yi*ky + (xi*kx)^2 + 2*xi*kx*b + b^2\n
    ====================================================================================================================\n
    d Σei^2 /dkx = Σ-zi*xi + ky*xi*yi + kx*xi^2 + xi*b = 0\n
    d Σei^2 /dky = Σ-zi*yi + ky*yi^2 + kx*xi*yi + b*yi = 0\n
    d Σei^2 /db  = Σ-zi + yi*ky + xi*kx = 0\n
    ====================================================================================================================\n
    d Σei^2 /dkx / dkx = Σ xi^2\n
    d Σei^2 /dkx / dky = Σ xi*yi\n
    d Σei^2 /dkx / db  = Σ xi\n
    ====================================================================================================================\n
    d Σei^2 /dky / dkx = Σ xi*yi\n
    d Σei^2 /dky / dky = Σ yi^2\n
    d Σei^2 /dky / db  = Σ yi\n
    ====================================================================================================================\n
    d Σei^2 /db / dkx = Σ xi\n
    d Σei^2 /db / dky = Σ yi\n
    d Σei^2 /db / db  = n\n
    ====================================================================================================================\n
    Hesse matrix:\n
    || d Σei^2 /dkx / dkx;  d Σei^2 /dkx / dky;  d Σei^2 /dkx / db ||\n
    || d Σei^2 /dky / dkx;  d Σei^2 /dky / dky;  d Σei^2 /dky / db ||\n
    || d Σei^2 /db  / dkx;  d Σei^2 /db  / dky;  d Σei^2 /db  / db ||\n
    ====================================================================================================================\n
    Hesse matrix:\n
                   | Σ xi^2;  Σ xi*yi; Σ xi |\n
    H(kx, ky, b) = | Σ xi*yi; Σ yi^2;  Σ yi |\n
                   | Σ xi;    Σ yi;    n    |\n
    ====================================================================================================================\n
                      | Σ-zi*xi + ky*xi*yi + kx*xi^2 + xi*b |\n
    grad(kx, ky, b) = | Σ-zi*yi + ky*yi^2 + kx*xi*yi + b*yi |\n
                      | Σ-zi + yi*ky + xi*kx                |\n
    ====================================================================================================================\n
    Окончательно решение:\n
    |kx|   |1|\n
    |ky| = |1| -  H(1, 1, 0)^-1 * grad(1, 1, 0)\n
    | b|   |0|\n

    :param x: массив значений по x
    :param y: массив значений по y
    :param z: массив значений по z
    :returns: возвращает тройку (kx, ky, b), которая является решением задачи (Σ(zi - (yi * ky + xi * kx + b))^2)->min
    """
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_z = np.sum(z)
    sum_xy = np.sum(x * y)
    sum_xx = np.sum(x ** 2)
    sum_yy = np.sum(y ** 2)
    sum_zy = np.sum(z * y)
    sum_zx = np.sum(z * x)
    n = len(x)

    hesse = np.array([[sum_xx, sum_xy, sum_x],
                      [sum_xy, sum_yy, sum_y],
                      [sum_x, sum_y, n]])

    grad = np.array([sum_xy + sum_xx - sum_zx,
                     sum_yy + sum_xy - sum_zy,
                     sum_y + sum_x - sum_z])

    return np.array([1, 1, 0]) - np.linalg.inv(hesse) @ grad


def n_linear_regression(data_rows: np.ndarray) -> np.ndarray:
    """
    H_ij = Σx_i * x_j, i in [0, rows - 1] , j in [0, rows - 1]
    H_ij = Σx_i, j = rows i in [rows, :]
    H_ij = Σx_j, j in [:, rows], i = rows

           | Σkx * xi^2    + Σky * xi * yi + b * Σxi - Σzi * xi|\n
    grad = | Σkx * xi * yi + Σky * yi^2    + b * Σyi - Σzi * yi|\n
           | Σyi * ky      + Σxi * kx                - Σzi     |\n

    x_0 = [1,...1, 0] =>

           | Σ xi^2    + Σ xi * yi - Σzi * xi|\n
    grad = | Σ xi * yi + Σ yi^2    - Σzi * yi|\n
           | Σxi       + Σ yi      - Σzi     |\n

    :param data_rows:  состоит из строк вида: [x_0,x_1,...,x_n, f(x_0,x_1,...,x_n)]
    :return:
    """
    s_rows, s_cols = data_rows.shape

    hessian = np.zeros((s_cols, s_cols,), dtype=float)

    grad = np.zeros((s_cols,), dtype=float)

    x_0 = np.zeros((s_cols,), dtype=float)

    for row in range(s_cols - 1):
        x_0[row] = 1.0
        for col in range(row + 1):
            value = np.sum(data_rows[:, row] @ data_rows[:, col])
            hessian[row, col] = value
            hessian[col, row] = value

    for i in range(s_cols):
        value = np.sum(data_rows[:, i])
        hessian[i, s_cols - 1] = value
        hessian[s_cols - 1, i] = value

    hessian[s_cols - 1, s_cols - 1] = data_rows.shape[0]

    for row in range(s_cols - 1):
        grad[row] = np.sum(hessian[row, 0: s_cols - 1]) - np.dot(data_rows[:, s_cols - 1], data_rows[:, row])

    grad[s_cols - 1] = np.sum(data_rows[:, s_cols - 1])

    return x_0 - np.linalg.inv(hessian) @ grad


def poly_regression(x: np.ndarray, y: np.ndarray, order: int = 10) -> np.ndarray:
    m = np.zeros((order, order), dtype=float)
    last_row = np.zeros((order,), dtype=float)
    n = len(x)

    cached_x_1 = 1

    for row in range(order):
        last_row[row] = np.sum(y * cached_x_1) / n

        cached_x_2 = 1
        for col in range(row + 1):
            m[col][row] = m[row][col] = np.sum(cached_x_1 * cached_x_2) / n
            cached_x_2 *= x

        cached_x_1 *= x

    return inv(m) @ last_row


def polynom(x: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    :param x: массив значений по x\n
    :param b: массив коэффициентов полинома\n
    :returns: возвращает полином yi = Σxi^j*bj\n
    """
    result = b[0] + b[1] * x
    for i in range(2, b.size):
        result += b[i] * x ** i
    return result


def distance_field_test():
    """
    Функция проверки поля расстояний:\n
    1) Посчитать тестовыe x и y используя функцию test_data\n
    2) Задать интересующие нас диапазоны k и b (np.linspace...)\n
    3) Рассчитать поле расстояний (distance_field) и вывести в виде изображения.\n
    4) Проанализировать результат (смысл этой картинки в чём...)\n
    :return:
    """
    x, y = test_data()
    k_, b_ = linear_regression(x, y)
    print(f"y(x) = {k_:1.5} * x + {b_:1.5}")
    k = np.linspace(-1000, 1000, 128, dtype=float)
    b = np.linspace(-1000, 1000, 128, dtype=float)
    z = distance_field(x, y, k, b)
    plt.imshow(z, extent=[k.min(), k.max(), b.min(), b.max()])
    plt.plot(k_, b_, 'r*')
    plt.xlabel("k")
    plt.ylabel("b")
    plt.grid(True)
    plt.show()


def linear_reg_test():
    """
    Функция проверки работы метода линейной регрессии:\n
    1) Посчитать тестовыe x и y используя функцию test_data\n
    2) Получить с помошью linear_regression значения k и b\n
    3) Вывести на графике x и y в виде массива точек и построить\n
       регрессионную прямую вида: y = k*x + b\n
    :return:
    """
    x, y = test_data()
    k, b = linear_regression(x, y)
    print(f"y(x) = {k:1.5} * x + {b:1.5}")
    plt.plot([x[0], x[len(x) - 1]], [b, k + b], 'g')
    plt.plot(x, y, 'r.')
    plt.show()


def bi_linear_reg_test():
    """
    Функция проверки работы метода билинейной регрессии:\n
    1) Посчитать тестовыe x, y и z используя функцию test_data_2d\n
    2) Получить с помошью bi_linear_regression значения kx, ky и b\n
    3) Вывести на трёхмерном графике x, y и z в виде массива точек и построить\n
       регрессионную плоскость вида:\n z = kx*x + ky*y + b\n
    :return:
    """
    x, y, z = test_data_2d()
    kx, ky, b = bi_linear_regression(x, y, z)
    print(f"z(x, y) = {kx:1.5} * x + {ky:1.5} * y + {b:1.5}")

    # x, y, z = test_data_nd().T
    # kx, ky, b = n_linear_regression(np.array([x, y, z])).T

    x_, y_ = np.meshgrid(np.linspace(np.min(x), np.max(x), 100), np.linspace(np.min(y), np.max(y), 100))
    z_ = kx * x_ + y_ * ky + b
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot(x, y, z, 'r.')
    surf = ax.plot_surface(x_, y_, z_, cmap=cm.coolwarm, linewidth=0, antialiased=False, edgecolor='none', alpha=0.5)
    plt.xlabel("x")
    plt.ylabel("y")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


def quadratic_linear_regression(x, y, z):
    n = len(x)
    base_arrays = [x * x, x * y, y * y, x, y, np.array([1.0])]
    a = np.zeros((6, 6))
    b = np.zeros(6)
    for row in range(6):
        b[row] = (base_arrays[row] * z).sum() / n
        for col in range(row + 1):
            a[col][row] = a[row][col] = (base_arrays[row] * base_arrays[col]).sum() / n
    a[5][5] = n

    return inv(a) @ b


def quadratic_reg_test():
    x, y, z = test_data_quad()
    kx2, ky2, kxy, kx, ky, b = quadratic_linear_regression(x, y, z)
    print(f"z(x, y) = {kx:1.5} * x + {ky:1.5} * y + {b:1.5}")

    x_, y_ = np.meshgrid(np.linspace(np.min(x), np.max(x), 100), np.linspace(np.min(y), np.max(y), 100))
    z_ = kx2 * x_ ** 2 + ky2 * y_ ** 2 + kxy * x_ * y_ + kx * x_ + y_ * ky + b
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot(x, y, z, 'r.')
    surf = ax.plot_surface(x_, y_, z_, cmap=cm.coolwarm, linewidth=0, antialiased=False, edgecolor='none', alpha=0.5)
    plt.xlabel("x")
    plt.ylabel("y")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


def poly_reg_test():
    """
    Функция проверки работы метода полиномиальной регрессии:\n
    1) Посчитать тестовыe x, y используя функцию test_data\n
    2) Посчитать набор коэффициентов bi полинома y = Σx^i*bi используя функцию poly_regression\n
    3) Вывести на графике x и y в виде массива точек и построить\n
       регрессионную кривую. Для построения кривой использовать метод polynom\n
    :return:
    """
    x, y = test_data(rand_range=1000, b=5)
    coefficients = poly_regression(x, y)
    y_ = polynom(x, coefficients)
    print(f"y(x) = {' + '.join(f'{coefficients[i]:.4} * x^{i}' for i in range(coefficients.size))}")
    plt.plot(x, y_, 'g')
    plt.plot(x, y, 'r.')
    plt.show()


def test_data_nd(surf_settings: np.ndarray = np.array([1.0, 1.0, 1.0]), vals_range: float = 1.0,
                 half_disp: float = 0.05, n_pts: int = 5) -> \
        np.ndarray:
    """
    Генерирует плоскость вида z = kx*x + ky*x + b + dz, где dz - аддитивный шум с амплитудой half_disp
    :param kx: наклон плоскости по x
    :param ky: наклон плоскости по y
    :param b: смещение по z
    :param half_disp: амплитуда разброса данных
    :param n: количество точек
    :param x_step: шаг между соседними точками по х
    :param y_step: шаг между соседними точками по y
    :returns: кортеж значенией по x, y и z
    """
    import random
    # surf_settings = [nx,ny,nz,d]

    data = np.zeros((n_pts, surf_settings.size,), dtype=float)

    for i in range(surf_settings.size - 1):
        data[:, i] = np.array([random.uniform(0.0, vals_range) for k in range(n_pts)])
        data[:, surf_settings.size - 1] += surf_settings[i] * data[:, i]

    dz = np.array([random.uniform(-half_disp, half_disp) for i in range(n_pts)])

    data[:, surf_settings.size - 1] += surf_settings[surf_settings.size - 1] + dz

    return data


if __name__ == "__main__":
    # data = test_data_nd()
    # print(data)
    # print(n_linear_regression(test_data_nd()))
    # distance_field_test()
    # linear_reg_test()
    # bi_linear_reg_test()
    # poly_reg_test()
    quadratic_reg_test()
