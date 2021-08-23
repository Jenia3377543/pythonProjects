import numpy as np
from random import randrange

class TextureGenerator:
    def __init__(self, pcd):
        self.pcd = pcd
        self.points = np.asarray(pcd.points)

    def solid_texture(self):
        height, width = 40, 40
        r_color = np.random.choice(self.pcd.colors[0],1)[0]
        g_color = np.random.choice(self.pcd.colors[1],1)[0]
        b_color = np.random.choice(self.pcd.colors[2],1)[0]
        rand_color=(r_color,g_color,b_color)

        image = np.zeros((height,width,3), np.uint8)
        #noise = np.random.normal(255. / 2, 255. / 10, (height,width,3))

        #color = tuple((rand_color))
        image[:] = rand_color

        return image