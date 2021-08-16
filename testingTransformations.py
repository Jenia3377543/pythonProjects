import open3d as o3d
import numpy as np
import copy

vector = np.asarray([1, 1, 1])
matrix = np.asarray([
    [1, 200, 1],
    [1, 2, 1],
    [1, 2, 1],
    [1, 2, 1],
])

vectors = np.subtract(matrix[:,:],vector)

print("vectors: {vectors}".format(vectors=vectors))

indexes = np.argsort(np.linalg.norm(vectors,axis=1))
print(vectors[indexes])

init_pointA = np.asarray([0, 0, 0])
init_pointB = np.asarray([0, 1, 0])
init_pointC = np.asarray([1, 1, 0])
init_pointD = np.asarray([1, 0, 0])

# final_pointA = np.asarray([1, 1, 0])
# final_pointB = np.asarray([1, 1, 1])
# final_pointC = np.asarray([1, 0, 1])
# final_pointD = np.asarray([1, 0, 0])

final_pointA = np.asarray([0, 0, 0])
final_pointB = np.asarray([1, 0, 1])
final_pointC = np.asarray([2, 0, 1])
final_pointD = np.asarray([1, 0, 0])




initial_points = np.asarray([
                    init_pointA,init_pointB,
                    init_pointC,init_pointD
                    ])
points = np.asarray([
                    [0,0,0],[0, 1, 0],
                    [1, 1, 0],[1, 0, 0]
                    ])

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(points/255)
pcd.normals = o3d.utility.Vector3dVector(np.asarray([[1,1,1],[1,1,1],[1,1,1],[1,1,1]]))

final_points = np.asarray([final_pointA,final_pointB,final_pointC,final_pointD])

AD = np.linalg.norm(np.subtract(final_pointA,final_pointD))
AB = np.linalg.norm(np.subtract(final_pointA,final_pointB))

S = np.asarray([
                [AD, 0 ,  0, 0],
                [ 0, AB,  0, 0],
                [ 0, 0 ,  1, 0],
                [ 0, 0 ,  0, 1],
                ])
X = np.subtract(final_pointD,final_pointA)/AD
Y = np.subtract(final_pointB,final_pointA)/AB
Z = np.cross(X,Y)

R = [
    [X[0], Y[0], Z[0], 0 ],
    [X[1], Y[1], Z[1], 0 ],
    [X[2], Y[2], Z[2], 0 ],
    [  0,   0,   0, 1 ],
]

O = np.mean(final_points,axis=0)

print("Translation: {translation}".format(translation=O))
initial_center_of_mass = np.mean(initial_points,axis=0)
T = [
    [1, 0, 0, O[0]-initial_center_of_mass[0]],
    [0, 1, 0, O[1]-initial_center_of_mass[1]],
    [0, 0, 1, O[2]-initial_center_of_mass[2]],
    [0, 0, 0,   1],
]


# M = np.matmul(np.matmul(T,R),S)
M = np.matmul(T,np.matmul(R,S))

print(M)

mesh_box = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=0.001)
mesh_box.compute_vertex_normals()
mesh_box.translate(-initial_center_of_mass)
mesh_box.paint_uniform_color([0.9, 0.1, 0.1])

initial_normal = [0, 0, 1]
normal = [1, 0, 0]

# o3d.io.write_triangle_mesh("./mesh_box.obj", mesh_box)

mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
# mesh.tr
mesh_box_copy = copy.deepcopy(mesh_box).transform(M)
mesh_box_copy.paint_uniform_color([0.7, 0.1, 0.9])

print("Previous center in : {center}".format(center=mesh_box.get_center()))
print("New center in : {center}".format(center=mesh_box_copy.get_center()))

o3d.visualization.draw_geometries([mesh, mesh_box_copy,mesh_box,pcd])


# x_1 = [93,-7]
# y_1 = [63,0]
#
# x_2 = [293,3]
# y_2 = [868,-6]
#
# x_3 = [1207,7]
# y_3 = [998,-4]
#
# x_4 = [1218,3]
# y_4 = [309,2]
#
# P = np.array([
#     [-x_1[0], -y_1[0], -1, 0, 0, 0, x_1[0]*x_1[1], y_1[0]*x_1[1], x_1[1]],
#     [0, 0, 0, -x_1[0], -y_1[0], -1, x_1[0]*y_1[1], y_1[0]*y_1[1], y_1[1]],
#     [-x_2[0], -y_2[0], -1, 0, 0, 0, x_2[0]*x_2[1], y_2[0]*x_2[1], x_2[1]],
#     [0, 0, 0, -x_2[0], -y_2[0], -1, x_2[0]*y_2[1], y_2[0]*y_2[1], y_2[1]],
#     [-x_3[0], -y_3[0], -1, 0, 0, 0, x_3[0]*x_3[1], y_3[0]*x_3[1], x_3[1]],
#     [0, 0, 0, -x_3[0], -y_3[0], -1, x_3[0]*y_3[1], y_3[0]*y_3[1], y_3[1]],
#     [-x_4[0], -y_4[0], -1, 0, 0, 0, x_4[0]*x_4[1], y_4[0]*x_4[1], x_4[1]],
#     [0, 0, 0, -x_4[0], -y_4[0], -1, x_4[0]*y_4[1], y_4[0]*y_4[1], y_4[1]],
#     ])
#
# [U, S, Vt] = np.linalg.svd(P)
# homography = Vt[-1].reshape(3, 3)
# transformedPoint = homography @ np.array([998,  -4, 1]).transpose()
# print(transformedPoint/transformedPoint[-1]) # will be ~[4, 7, 1]
