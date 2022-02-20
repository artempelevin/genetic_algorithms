from typing import Optional, Callable, Any

import numpy as np


def f_(x: Any) -> Any:
    return x * x


def tournament_selection(population: np.array, f: Callable, n: int, k: Optional[int] = 2) -> np.array:
    """
    Турнирный отбор. Отбираются k-хромосом n-раз и в каждом таком турнире выбирается лучшая хромосома.
    :param population: Текущая популяция, представленная массивом хромосом.
    :param f: Фитнес-функция, благодаря которой выбирается лучшая хромосома в турнире.
    :param n: Кол-во проводимых турниров.
    :param k: Кол-во хромосом, участвующих в турнире (по умолчанию 2).
    :return: Новая популяция, состоящая из n-элементов - победителей турниров.
    """
    new_population = []
    for _ in range(n):
        random_chromosomes = np.random.choice(population, size=k)
        values = [f(x) for x in random_chromosomes]
        min_value = min(values)
        new_population.append(random_chromosomes[values.index(min_value)])

    return np.array(new_population)


def truncation_selection(population: np.array, f: Callable, l: float) -> np.array:
    """
    Селекция усечением. Отбираются ln лучших хромосом, где l-порог отсечения.
    :param population: Текущая популяция, представленная массивом хромосом.
    :param f: Фитнес-функция, благодаря которой выбирается лучшая хромосома в турнире.
    :param l: Порог отсечения в диапазоне 0 < l < 1. Чем меньше l, тем сильнее давление селекции.
    :return: Новая популяция, состоящая из n-элементов - победителей турниров.
    """
    assert (0 < l < 1)
    f_ = np.vectorize(f)
    values = f_(population)
    return np.sort(np.dstack((population, values)), axis=1)[0, :int(population.size * l), 0]


if __name__ == '__main__':
    population_ = np.array([10, 12, 3, 4, 5])
    print(truncation_selection(population=population_, f=f_, l=0.5))
