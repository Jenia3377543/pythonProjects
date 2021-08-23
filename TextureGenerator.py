import numpy as np

class TextureGenerator:
    def __init__(self, pcd):
        self.pcd = pcd
        self.points = np.asarray(pcd.points)

    def solid_texture(self):
        height, width = 40, 40
        avg_color = np.mean(self.points, axis=0)
        image = np.zeros((height,width,3), np.uint8)
        noise = np.random.normal(255. / 2, 255. / 10, (height,width,3))

        color = tuple(reversed(avg_color))
        image[:] = color+noise

        return image