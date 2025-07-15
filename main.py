from make_hash import Hash, res_color

from math import gcd
from functools import reduce
import numpy as np
import os

import matplotlib.pyplot as plt

from scipy.optimize import lsq_linear


def mix_colors(colors_array):
    """Функция для определения пропорций смешивания светов с учетом яркости"""
    normalized = colors_array / 255.0

    # Отделяем источники от целевого света
    sources = normalized[:-1, :3].T  # RGB исходных светов
    target = normalized[-1, :3]  # RGB целевого света
    target_brightness = normalized[-1, 3] # Яркость целевого света

    # Максимальные яркости источников
    max_brightness = normalized[:-1, 3]

    # Решаем систему методом наименьших квадратов с ограничениями на коэффициенты
    bounds = (np.zeros_like(max_brightness), max_brightness)
    result = lsq_linear(sources, target, bounds=bounds)

    # Получаем и нормализуем коэффициенты
    coefficients = result.x

    # Масштабируем коэффициенты так, чтобы сумма яркостей = яркости целевого цвета
    if np.sum(coefficients) > 0:
        scale_factor = target_brightness / np.sum(coefficients)
        real_brightness = coefficients * scale_factor
    else:
        real_brightness = np.zeros_like(coefficients)

    # Нормализуем пропорции (сумма = 1)
    proportions = real_brightness / target_brightness if target_brightness > 0 else np.zeros_like(
        real_brightness)

    return proportions, real_brightness * 255.0


def apply_alpha(color, background=(255, 255, 255)):
    """Применяет альфа-канал к свету на заданном фоне"""
    alpha = color[3] / 255.0
    return (
        int(color[0] * alpha + background[0] * (1 - alpha)),
        int(color[1] * alpha + background[1] * (1 - alpha)),
        int(color[2] * alpha + background[2] * (1 - alpha))
    )


def color_similarity(original_color, mixed_color):
    """Вычисляет процент совпадения светов (0-100)"""
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
    """Смешивает света в заданных пропорциях (работает с целыми числами)"""
    ratios = np.array(ratios)
    mixed_color = np.sum(colors_array[:-1] * ratios[:, np.newaxis], axis=0) / np.sum(ratios)
    return mixed_color


def optimal_similarity(all_ratios, similarities, threshold=0.1):
    """Поиск самого подходящего варианта с учётом погрешности - по сути, функция потерь"""
    max_similarity = max(similarities)

    # Фильтруем соотношения, у которых точность "достаточно близка" к максимальной
    valid_indices = [i for i, sim in enumerate(similarities)
                     if sim >= max_similarity - threshold]

    if not valid_indices:
        return None  # Нет подходящих вариантов

    # Выбираем среди них соотношение с минимальной суммой чисел
    valid_ratios = [all_ratios[i] for i in valid_indices]
    sums = [sum(ratio) for ratio in valid_ratios]
    optimal_idx = np.argmin(sums)

    return valid_ratios[optimal_idx]


def plot_color_matching_accuracy(percentages, colors_array, max_precision=1000, step=10):
    """Строит график зависимости точности смешивания от параметра precision"""
    target_color = colors_array[-1]

    precisions = range(1, max_precision + 2, step)
    similarities = []
    all_ratios = []

    for precision in precisions:
        # Преобразуем проценты в целые соотношения
        ratios = float_to_ratio(percentages, precision)
        all_ratios.append(list(map(int, ratios)))

        # Смешиваем света
        mixed_color = mix_colors_with_ratio(colors_array, ratios)

        # Сравниваем с целевым
        similarity = color_similarity(target_color, mixed_color)
        similarities.append(similarity)

    # Находим самое большое совпадение и выводим его коэффициенты как ответ
    best_similarity = max(similarities)
    print(f'\nСамая большая точность: {best_similarity} %')
    needed_lights = all_ratios[similarities.index(best_similarity)]
    print("Для её достижения нужны:")
    for cou, light in enumerate(needed_lights):
        print(f"Свет {cou + 1}: {light}")

    # А теперь находим самое большое совпадение с учётом возможной погрешности
    needed_lights = optimal_similarity(all_ratios, similarities)
    best_similarity = similarities[all_ratios.index(needed_lights)]
    print(f'\nСамая оптимальная точность: {best_similarity} %')
    print('Для её достижения нужны:')
    for cou, light in enumerate(needed_lights):
        print(f"Свет {cou + 1}: {light}")

    # Построение графика (опционально, но я решил сделать)
    plt.figure(figsize=(10, 6))
    plt.plot(precisions, similarities, label="Совпадение с целевым светом", color='blue')
    plt.xlabel("Precision (точность округления)")
    plt.ylabel("Совпадение света (%)")
    plt.title("Зависимость качества смешивания от соотношения светов")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Подгружаем света
    [res_color(Hash(f'templates/{name}').get_grey(), f'{name[:-4]}.png') for name in
     os.listdir(os.path.join('templates'))]
    my_list = np.array([Hash(f'templates/{name}').get_grey() for name in os.listdir(os.path.join('templates'))])

    # Получаем ключевые данные
    proportions, brightnesses = mix_colors(my_list)

    # Красивый вывод и график
    print("Пропорции смешивания:")
    for i, prop in enumerate(proportions):
        print(f"Свет {i + 1}: {prop * 100:.5f}%")

    print("\nНеобходимые яркости:")
    for i, bri in enumerate(brightnesses):
        print(f"Свет {i + 1}: {bri:.2f}")
    plot_color_matching_accuracy(proportions, my_list, max_precision=1000, step=10)