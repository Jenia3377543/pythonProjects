import open3d as o3d
import numpy as np
def plane_intersect(a, b):
    """
    a, b   4-tuples/lists
           Ax + By +Cz + D = 0
           A,B,C,D in order

    output: 2 points on line of intersection, np.arrays, shape (3,)
    """
    a_vec, b_vec = np.array(a[:3]), np.array(b[:3])

    aXb_vec = np.cross(a_vec, b_vec)

    A = np.array([a_vec, b_vec, aXb_vec])
    d = np.array([-a[3], -b[3], 0.]).reshape(3,1)

# could add np.linalg.det(A) == 0 test to prevent linalg.solve throwing error

    p_inter = np.linalg.solve(A, d).T

    return p_inter[0], (p_inter + aXb_vec)[0]




input_path="C:/Technion/semester6/project/pythonProjects/"
output_path="C:/Technion/semester6/project/pythonProjects/outputs"


datanameply="block_result.ply"
mesh = o3d.io.read_triangle_mesh(input_path+datanameply)
point_cloud_sampled = mesh.sample_points_poisson_disk(5000)

points = np.asarray(point_cloud_sampled.points)
min_height = np.min(points[:,2])
max_height = np.max(points[:,2])

diff = max_height - min_height
segments_count = 1
segment_height = diff/5

print(diff)
# new_points = points[(points[:,2]>(min_height+segment_height*3)) & (points[:,2]<(min_height+segment_height*4))]
new_points = points

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(new_points[:,:3])
pcd.colors = o3d.utility.Vector3dVector(new_points[:,0:3]/255)
pcd.normals = o3d.utility.Vector3dVector(new_points[:,0:3])

mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=15, origin=np.mean(points,axis=0))

distances = pcd.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)
radius = 1.5 * avg_dist

objects = list()

min_distance = np.min(distances)
labels = np.asarray(pcd.cluster_dbscan(eps=6*min_distance, min_points=50))
print(np.unique(labels))

for label in np.unique(labels):
    object = o3d.geometry.PointCloud()
    object.points = o3d.utility.Vector3dVector(new_points[labels==label, :3])
    object.colors = o3d.utility.Vector3dVector(new_points[labels==label, 0:3] / 255)
    object.normals = o3d.utility.Vector3dVector(new_points[labels==label, 0:3])

    objects.append(object)

planes = list()
objects_planes_normals = list()

for idx, object in enumerate(objects):
    planes.append(list())
    objects_planes_normals.append(list())
    while(True):
        plane_model, inliers = object.segment_plane(distance_threshold=0.01, ransac_n=4, num_iterations=1000)

        inlier_cloud = object.select_by_index(inliers)
        rand_colors = np.abs(np.random.randn(3,1))

        inlier_cloud.paint_uniform_color(rand_colors)
        planes[idx].append(inlier_cloud)
        objects_planes_normals[idx].append(plane_model)

        object = object.select_by_index(inliers, invert=True)

        if(len(np.asarray(object.points)) < 10):
            break

threshold = 5
planes_connectivity = list()
for obj_ind, object in enumerate(planes):
    planes_connectivity.append(list())
    for ind1, plane1 in enumerate(planes[obj_ind]):
        planes_connectivity[obj_ind].append(list())
        for ind2, plane2 in enumerate(planes[obj_ind]):
            if(ind1 == ind2):
                continue

            distance = np.min(plane1.compute_point_cloud_distance(plane2))
            if(distance < threshold):
                planes_connectivity[obj_ind][ind1].append(ind2)

print(planes_connectivity)
print(len(planes))
print('objects planes normals=',objects_planes_normals)
# o3d.visualization.draw_geometries(np.asarray(planes))
# o3d.visualization.draw_geometries([mesh])

# print(plane_model)

object_planes_points = list()
for idx, object_connectivity_list in enumerate(planes_connectivity):
    object_planes_points.append(list())
    for conn_index, plane_connectivity_list in enumerate(object_connectivity_list):
        for connection in plane_connectivity_list:
            common_planes = np.intersect1d(plane_connectivity_list,object_connectivity_list[connection])

            for common_item in common_planes:
                object_planes_points[idx].append([conn_index,connection,common_item])

print(object_planes_points)
object_planes_points2_sorted = list()
for indx, object in enumerate(object_planes_points):
    object_planes_points2_sorted.append(list())
    for triple in object:
        object_planes_points2_sorted[indx].append(np.sort(triple))

object_planes_points_distinct_and_sorted = list()

for object in object_planes_points2_sorted:
    object_planes_points_distinct_and_sorted.append(np.unique(object,axis=0))

print(object_planes_points_distinct_and_sorted)

intersection_point = list()

for idx, object in enumerate(object_planes_points_distinct_and_sorted):

    for triple in object:

        plane1 = objects_planes_normals[idx][triple[0]]
        plane2 = objects_planes_normals[idx][triple[1]]
        plane3 = objects_planes_normals[idx][triple[2]]

        point11, point12 = plane_intersect(plane1, plane2)
        point21, point22 = plane_intersect(plane2, plane3)

        vector1 = np.subtract(point12,point11)
        vector2 = np.subtract(point22,point21)

        point = np.cross(vector1,vector2)
        norm = np.sqrt(np.dot(point,point))

        point = point/norm
        intersection_point.append(point)
        print(point)

intersection_point = np.asarray(intersection_point)
finalObject = o3d.geometry.PointCloud()
finalObject.points = o3d.utility.Vector3dVector(intersection_point[:, :3])
finalObject.colors = o3d.utility.Vector3dVector(intersection_point[:, 0:3] / 255)
finalObject.normals = o3d.utility.Vector3dVector(intersection_point[:, 0:3])
o3d.visualization.draw_geometries([finalObject])

meshes = list()

for object in planes:
    for plane in object:
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                   plane,
                   o3d.utility.DoubleVector([radius, radius * 2]))
        mesh.compute_vertex_normals()
        # mesh.compute_triangle_normals()
        meshes.append(mesh)

print(len(meshes))

o3d.visualization.draw_geometries(np.asarray(meshes))

# o3d.visualization.draw_geometries(np.asarray(meshes))
# bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,o3d.utility.DoubleVector([radius, radius * 1.5]))
# bpa_mesh.compute_triangle_normals()
# o3d.visualization.draw_geometries([mesh2])
# o3d.io.write_triangle_mesh(output_path+"bpa_mesh11.ply", total_mesh,write_vertex_colors=False,write_vertex_normals=False,write_triangle_uvs=False,write_ascii=True)

print("Let's draw a box using o3d.geometry.LineSet.")
points = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
]
lines = [
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 3],
    [4, 5],
    [4, 6],
    [5, 7],
    [6, 7],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
]
colors = [[1, 0, 0] for i in range(len(lines))]
line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(points),
    lines=o3d.utility.Vector2iVector(lines),
)
line_set.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([line_set])

