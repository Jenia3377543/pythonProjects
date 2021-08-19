import open3d as o3d
import numpy as np
import copy
print(o3d.__version__)

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

def euclidean_transform_3D(A, B):
    '''
        A,B - Nx3 matrix
        return:
            R - 3x3 rotation matrix
            t = 3x1 column vector
    '''
    assert len(A) == len(B)

    # number of points
    N = A.shape[0]

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # centre matrices
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    # covariance of datasets
    H = np.transpose(AA) * BB

    # matrix decomposition on rotation, scaling and rotation matrices
    U, S, Vt = np.linalg.svd(H)

    # resulting rotation
    R = Vt.T * U.T
    print('R', R)
    # prinyt(Vt)
    print(Vt)
    # handle svd sign problem
    if np.linalg.det(R) < 0:
        print("sign")
        # thanks to @valeriy.krygin to pointing me on a bug here
        Vt[2, :] *= -1
        R = Vt.T * U.T
        print('new R', R)

    t = -R * centroid_A.T + centroid_B.T

    return R, t, S

total_final_points = np.asarray([
    [
    [-427.54276195, -365.82275331,   36.65242971],
    [-382.88294596, -408.37695951,   36.66888847],
    [-444.28824362, -382.86324763,   36.79383354],
    [-399.97161631, -425.71761334,   36.81297079]
    ],

    [
    [-435.91728389, -374.34436792,   51.36386161],
    [-391.57436084, -417.19639834,   51.31907956],
    [-427.54396208, -365.82373641,   44.48754405],
    [-382.88401967, -408.37797857,   44.19267638]
    ],
    [
    [-435.91728389, -374.34436792,   51.36386161],
    [-391.57436084, -417.19639834,   51.31907956],
    [-444.2877768 , -382.86253979,   44.4523736 ],
    [-399.97107296, -425.71699094,   44.39605161]
    ],
    [
    [-427.54276195, -365.82275331,   36.65242971],
    [-382.88294596, -408.37695951,   36.66888847],
    [-427.54396208, -365.82373641,   44.48754405],
    [-382.88401967, -408.37797857,   44.19267638]
    ]
])

total_initial_points = np.asarray([
    [
    [0, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [1, 0, 0]
    ],

    [
        [0, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [1, 0, 0]
    ],
    [
        [0, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [1, 0, 0]
    ],
    [
        [0, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [1, 0, 0]
    ]
])
planes = list()
total_points = list()

for idx, final_points in enumerate(total_final_points):
    for point in final_points:
        total_points.append(point)
    vectors = np.subtract(final_points[1:,:],final_points[0])

    indexes = np.argsort(np.linalg.norm(vectors,axis=1))
    vector1 = vectors[indexes[1]]
    vector2 = vectors[indexes[0]]

    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    print("norm1={norm1},norm2={norm2}".format(norm1=norm1,norm2=norm2))

    S = np.asarray([
                    [norm1, 0 ,  0, 0],
                    [ 0, norm2,  0, 0],
                    [ 0, 0 ,   1, 0],
                    [ 0, 0 ,  0, 1],
                    ])
    X = vector1/norm1
    Y = vector2/norm2
    Z = np.cross(X,Y)

    R = [
        [X[0], Y[0], Z[0], 0 ],
        [X[1], Y[1], Z[1], 0 ],
        [X[2], Y[2], Z[2], 0 ],
        [  0,   0,   0, 1 ],
    ]

    O = np.mean(final_points,axis=0)

    print("Translation: {translation}".format(translation=O))

    T = [
        [1, 0, 0, O[0]],
        [0, 1, 0, O[1]],
        [0, 0, 1, O[2]],
        [0, 0, 0,   1],
    ]

    M = np.matmul(np.matmul(T,R),S)


    # R, t, S = euclidean_transform_3D(np.mat(total_initial_points[idx]), np.mat(final_points))
    # R = np.asarray(R)
    # S = np.asarray(S)
    # print(f"rotation matrix: {R}")
    # print(f"scale matrix: {S}")
    # S = [
    #     [1,     0 ,    0, 0],
    #     [0   ,  1 ,    0, 0],
    #     [0   ,  0 ,    1, 0],
    #     [0,     0 ,    0, 1]
    #                 ]
    #
    # R = [
    #     [R[0][0],R[0][1],R[0][2],0],
    #     [R[1][0],R[1][1],R[1][2],0],
    #     [R[2][0],R[2][1],R[2][2],0],
    #     [0,0,0,1]
    #                 ]
    #
    # T = [
    #     [1, 0, 0, t[0]],
    #     [0, 1, 0, t[1]],
    #     [0, 0, 1, t[2]],
    #     [0, 0, 0,   1],
    # ]
    #
    # M = np.matmul(np.matmul(T, R),np.asarray(S))

    print(M)

    mesh_box = o3d.geometry.TriangleMesh.create_box(width=1., height=1, depth=0.001,create_uv_map=True,map_texture_to_each_face=True)

    mesh_box.compute_vertex_normals()
    # mesh_box.paint_uniform_color([0.9, 0.1, 0.1])

    mesh_box_copy = copy.deepcopy(mesh_box).translate(-mesh_box.get_center()).transform(M)
    # mesh_box_copy = copy.deepcopy(mesh_box).rotate(R).translate(t)
    # mesh_box_copy = copy.deepcopy(mesh_box).transform(M)
    # mesh_box_copy.paint_uniform_color([0.7, 0.1, 0.9])
    planes.append(mesh_box_copy)
# o3d.io.write_triangle_mesh("./mesh_box.obj", mesh_box)

total_points = np.asarray(total_points)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(total_points)
# pcd.colors = o3d.utility.Vector3dVector(total_final_points / 1000)
# pcd.normals = o3d.utility.Vector3dVector(total_final_points)

mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()

o3d.visualization.draw_geometries([np.sum(np.asarray(planes)), pcd])


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
