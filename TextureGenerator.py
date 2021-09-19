import numpy as np

class TextureTail:
    def __init__(self, material, size):
        self.material_tail = material
        self.real_width = size[0]
        self.real_height = size[1]

def tailed_texture(points, tail):
    print(f"points = {points}")
    distances = np.sort(np.linalg.norm(np.subtract(points, points[0]),axis=1),axis=None)
    print(f"distances = {distances}")

    height = distances[1]
    width = distances[2]

    tail_heigth, tail_width, channels = tail.material_tail.shape

    print(f"tail_heigth = {tail_heigth}; tail_width = {tail_width}; ")
    scale_heigth = int(height/tail.real_height)
    scale_width = int(width/tail.real_width)

    print(f"scale_heigth={scale_heigth}, scale_width={scale_width}")
    image = np.zeros((tail_heigth * scale_heigth, scale_width * tail_width, 3), np.uint8)

    for col in range(scale_width):
        for row in range(scale_heigth):
            image[tail_heigth * row:tail_heigth * (row + 1), col * tail_width:(col + 1) * tail_width, :] = tail.material_tail

    return image

class TextureGenerator:
    def __init__(self, pcd):
        self.pcd = pcd
        self.points = np.asarray(pcd.points)

    def solid_texture(self):
        height, width = 40, 40
        avg_color = np.mean(self.points, axis=0)
        image = np.zeros((height,width,3), np.uint8)

        color = tuple(reversed(avg_color))
        image[:] = color

        return image

