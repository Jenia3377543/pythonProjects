import numpy as np
import cv2
import open3d as o3d

def points_to_triangles(ordered_points):
    initial_point = ordered_points[0]
    points_count = len(ordered_points)

    triangles_list = list()
    start_idx = 1
    while(start_idx+1<points_count):
        triangles_list.append([initial_point,
                               ordered_points[start_idx],
                               ordered_points[start_idx+1]])
        start_idx = start_idx + 1
    return np.asarray(triangles_list)


def order_clockwise2(points, normal=[0,0,0]):
    center_of_mass = np.mean(points,axis=0)
    print(f"center of mass: {center_of_mass}")
    possible_axis = np.subtract(points,points[0])
    print(f"possible axis: {possible_axis}")
    p = possible_axis[np.argsort(np.linalg.norm(possible_axis,axis=1))[-1]]
    print(f"p: {p}")
    q = np.cross(normal,p)
    print(f"q: {q}")

    indexes = np.argsort(
        np.arctan2(
            np.dot(np.cross(np.subtract(points, center_of_mass), q),normal),
            np.dot(np.cross(np.subtract(points, center_of_mass), p),normal)
        )
    )
    print(f"indexes: {indexes}")
    return points[indexes]


vert=[
        [0,0,0],[0,1,0],[2,1,0],[2,0,0],
        [0,0,1],[0,1,1],[2,1,1],[2,0,1]
]

points = [ [0,1,0],[2,0,0],[0,0,0],[2,1,0] ]
matrix = np.asarray(
    [
        [1, 2, 3],
        [4, 5, 6],
        [3, 2, 2],
    ]
)

ordered_points = order_clockwise2(np.asarray(points),[0,0,-1])

print(f"ordered points {ordered_points}")
triangles = points_to_triangles(ordered_points)
print(f"triangles list(): {triangles}")

new_faces = [[np.where(np.all(np.asarray(vert)==point,axis=1))[0][0] for point in triangle] for triangle in triangles]
print(f"triangles list by indexes: {new_faces}")
# faces=[
#     [0, 1, 2], [0, 2, 3], [6, 5, 4],
#     [7, 6, 4], [5, 1, 0], [0, 4, 5],
#     [3, 2, 6], [6, 7, 3], [0, 3, 7],
#     [0, 7, 4], [1, 5, 6], [1, 6, 2]]

faces=[
    [0, 1, 2], [2, 3, 0], [0, 4, 5],
    [5, 1, 0], [1, 5, 6], [6, 2, 1],
    [6, 7, 3], [3, 2, 6], [7, 4, 0],
    [0, 3, 7], [6, 5, 4], [4, 7, 6]]

m=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vert),
                            o3d.utility.Vector3iVector(faces))

text = cv2.imread('./cube_uv.png')
cv2.imshow('Cube map - uv',text)

m.compute_vertex_normals()
o3d.visualization.draw_geometries([m])

DX,DY=0.25, 0.33

v_uv=[
        [DX,DY],
        [DX,2*DY],
        [2*DX,2*DY],

        [2 * DX, 2 * DY],
        [2*DX,DY],
        [DX,DY],

        [DX, DY],
        [DX, 2 * DY],
        [2 * DX, 2 * DY],

        [2 * DX, 2 * DY],
        [2 * DX, DY],
        [DX, DY],
      ]

v_uv=np.asarray(v_uv)
v_uv=np.concatenate((v_uv,v_uv,v_uv),axis=0)

# v_uv = np.random.rand(len(faces) * 3, 2)

m.triangle_uvs = o3d.utility.Vector2dVector(v_uv)
m.textures=[o3d.geometry.Image(text)]
m.triangle_material_ids = o3d.utility.IntVector([0]*len(faces))

o3d.visualization.draw_geometries([m])