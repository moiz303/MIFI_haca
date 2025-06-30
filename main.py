from make_hash import Hash, res_color
from scipy.optimize import lsq_linear
from math import gcd
from functools import reduce
import numpy as np
import os

import matplotlib.pyplot as plt


def mix_colors(colors_array):
    """Функция для определения пропорций смешивания произвольного количества цветов для получения целевого цвета"""
    source_colors = colors_array[:-1].T  # Транспонируем для правильной формы матрицы
    target_color = colors_array[-1]

    # Решаем систему методом наименьших квадратов с ограничением: коэффициенты между 0 и 1
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


def color_similarity(original_color, mixed_color):
    """Вычисляет процент совпадения цветов (0-100%)"""
    diff = np.sqrt(np.sum((original_color - mixed_color) ** 2))  # Евклидово расстояние
    max_diff = np.sqrt(3 * 255 ** 2)  # Максимальное возможное расстояние (чёрный и белый)
    similarity = 100 * (1 - diff / max_diff)
    return similarity


def float_to_ratio(percentages, precision=1000):
    """Преобразует массив процентных соотношений в соотношения простых чисел"""
    scaled = np.round(np.array(percentages) * precision).astype(int)

    # Находим НОД для всех чисел
    def find_gcd(list_numbers):
        return reduce(lambda x, y: gcd(x, y), list_numbers)

    common_divisor = find_gcd(scaled)
    # Делим на НОД и возвращаем
    simple_ratio = scaled // common_divisor
    return simple_ratio


def mix_colors_with_ratio(colors_array, ratios):
    """Смешивает цвета в заданных пропорциях (работает с целыми числами)"""
    ratios = np.array(ratios)
    mixed_color = np.sum(colors_array[:-1] * ratios[:, np.newaxis], axis=0) / np.sum(ratios)
    return mixed_color


def plot_color_matching_accuracy(colors_array, max_precision=1000, step=10):
    """Строит график зависимости точности смешивания от параметра precision"""
    source_colors = colors_array[:-1]
    target_color = colors_array[-1]

    # Вычисляем идеальные проценты (через наименьшие квадраты)
    percentages = lsq_linear(source_colors.T, target_color, bounds=(0, 1)).x
    percentages /= percentages.sum()  # Нормализуем

    precisions = range(1, max_precision + 1, step)
    similarities = []

    for precision in precisions:
        # Преобразуем проценты в целые соотношения
        ratios = float_to_ratio(percentages, precision)

        # Смешиваем цвета
        mixed_color = mix_colors_with_ratio(colors_array, ratios)

        # Сравниваем с целевым
        similarity = color_similarity(target_color, mixed_color)
        similarities.append(similarity)

    # Построение графика
    plt.figure(figsize=(10, 6))
    plt.plot(precisions, similarities, label="Совпадение с целевым цветом", color='blue')
    plt.xlabel("Precision (точность округления)")
    plt.ylabel("Совпадение цвета (%)")
    plt.title("Зависимость качества смешивания от параметра precision")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    [res_color(Hash(f'templates/{name}').get_grey(), f'{name[:-4]}.png') for name in
     os.listdir(os.path.join('templates'))]
    my_list = np.array([Hash(f'templates/{name}').get_grey() for name in os.listdir(os.path.join('templates'))])
    proportions = mix_colors(my_list)
    print("Пропорции смешивания:")
    for i, prop in enumerate(proportions):
        print(f"Цвет {i + 1}: {prop * 100:.5f}%")
    plot_color_matching_accuracy(proportions, max_precision=1000, step=10)