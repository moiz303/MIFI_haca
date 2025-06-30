from make_hash import Hash, res_color
from scipy.optimize import nnls, lsq_linear
import numpy as np
import os

import matplotlib.pyplot as plt

[res_color(Hash(f'templates/{name}').get_grey(), f'{name[:-4]}.png') for name in os.listdir(os.path.join('templates'))]
my_list = np.array([Hash(f'templates/{name}').get_grey() for name in os.listdir(os.path.join('templates'))])


def mix_colors(colors_array):
    """Функция для определения пропорций смешивания произвольного количества цветов
    для получения целевого цвета"""
    source_colors = colors_array[:-1].T  # Транспонируем для правильной формы матрицы
    target_color = colors_array[-1]

    # Решаем систему методом наименьших квадратов с ограничениями
    # Ограничения: коэффициенты между 0 и 1
    result = lsq_linear(source_colors, target_color, bounds=(0, 1))

    # Нормализуем коэффициенты, чтобы их сумма была равна 1
    coefficients = result.x
    normalized_coefficients = coefficients / coefficients.sum()

    return normalized_coefficients


def apply_alpha(color, background=(255, 255, 255)):
    """Применяет альфа-канал к цвету на заданном фоне."""
    alpha = color[3] / 255.0
    return (
        int(color[0] * alpha + background[0] * (1 - alpha)),
        int(color[1] * alpha + background[1] * (1 - alpha)),
        int(color[2] * alpha + background[2] * (1 - alpha))
    )


if __name__ == "__main__":
    proportions = mix_colors(my_list)
    print("\nПропорции смешивания:")
    for i, prop in enumerate(proportions):
        print(f"Цвет {i + 1}: {prop * 100:.5f}%")