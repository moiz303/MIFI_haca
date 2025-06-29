from PIL import Image

class Hash:
    def __init__(self, name) -> None:
        img = Image.open(name)
        self.size = min(img.size)
        img2 = img.resize((self.size, self.size))
        self.pixels = img2.load()
        img.close()


    def get_grey(self) -> tuple:
        R, G, B, L = 0, 0, 0, 0
        for i in range(self.size):
            for j in range(self.size):
                r, g, b, l = self.pixels[i, j]
                R += r
                G += g
                B += b
                L += l
        return R/ self.size ** 2, G/ self.size ** 2, B/ self.size ** 2, L/ self.size ** 2
