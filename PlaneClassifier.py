import numpy as np

x_vector = np.asarray([1, 0, 0])
y_vector = np.asarray([0, 1, 0])
z_vector = np.asarray([0, 0, 1])

epsilon = np.pi/15

def classifyPlane(normal_vector):
    normal_vector_norm = np.linalg.norm(normal_vector)
    angle = np.arccos(np.dot(normal_vector, z_vector)/normal_vector_norm)

    if((angle >= np.pi/2 - epsilon) and (angle <= np.pi/2 + epsilon)):
        #WALL
        return 0
    elif(((angle >= -epsilon) and (angle <= epsilon)) or (angle >= np.pi-epsilon)):
        # FLOOR
        return 1
    else:
        # ROOF
        return 2