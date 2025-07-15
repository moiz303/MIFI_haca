from PIL import Image

class Hash:
    """Класс по преобразованию цвета в его цветовой код - база программы"""
    def __init__(self, name) -> None:
        img = Image.open(name)
        self.size = min(img.size)
        img2 = img.resize((self.size, self.size))
        self.pixels = img2.load()
        img.close()


    def get_grey(self) -> tuple:
        """Разбираем картинку на пиксели и берём среднее для R, G, B, и Альфы (яркости цвета)"""
        R, G, B, A = 0, 0, 0, 0
        for i in range(self.size):
            for j in range(self.size):
                r, g, b, a = self.pixels[i, j]
                R += r
                G += g
                B += b
                A += a
        return R/ self.size ** 2, G/ self.size ** 2, B/ self.size ** 2, A/ self.size ** 2


def res_color(colors: tuple, name: str):
    """Записывает в файл заливку  из заданного цвета"""
    img = Image.new('RGBA', [500, 500], tuple(map(int, colors)))
    img.save(name)
    img.close()


if __name__ == '__main__':
    first = Hash('templates/file.png').get_grey()
    second = Hash('templates/file2.png').get_grey()
    third = Hash('templates/file3.png').get_grey()
    fourth = Hash('templates/file4.png').get_grey()
    #res_color(fourth, 'file4.png')
    print(first, '\n', second, '\n', third, '\n', fourth)