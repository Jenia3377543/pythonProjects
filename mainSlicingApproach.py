import open3d as o3d
import numpy as np
import removeLines


input_path="C:/Technion/semester6/project/pythonProjects/"
output_path="C:/Technion/semester6/project/pythonProjects/outputs"
datanameply="block_result.ply"
dataname="block.off"

#--------------------------------------------- Getting points from mesh --------------------------------------
mesh = o3d.io.read_triangle_mesh(input_path+datanameply)
point_cloud_sampled = mesh.sample_points_poisson_disk(5000)

points = np.asarray(point_cloud_sampled.points)
min_height = np.min(points[:,2])
max_height = np.max(points[:,2])

diff = max_height - min_height
segments_count = 5
segment_height = diff/segments_count

print(diff)
# new_points = points[(points[:,2]>(min_height+segment_height*3)) & (points[:,2]<(min_height+segment_height*4))]
point_variance = 0
noise = np.random.normal(points,point_variance,points.shape)

new_points = points
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(new_points[:,:3])
pcd.colors = o3d.utility.Vector3dVector(new_points[:,0:3]/255)
pcd.normals = o3d.utility.Vector3dVector(new_points[:,0:3])
o3d.visualization.draw_geometries([pcd])

mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=15, origin=np.mean(points,axis=0))
#--------------------------------------------- Divide points to objects --------------------------------------
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
print("objects count : {objects}".format(objects=len(np.unique(labels))))

#--------------------------------------------- Fit planes with RANSAC --------------------------------------
planes = list()
objects_planes_normals = list()
for idx, object in enumerate(objects):
    planes.append(list())
    objects_planes_normals.append(list())

    while(True):
        plane_model, inliers = object.segment_plane(distance_threshold=0.01, ransac_n=4, num_iterations=1000)

        inlier_cloud = object.select_by_index(inliers)

        if(len(np.asarray(inlier_cloud.points)) > 30):
            rand_colors = np.abs(np.random.randn(3,1))
            inlier_cloud.paint_uniform_color(rand_colors)
            planes[idx].append(inlier_cloud)
            objects_planes_normals[idx].append(plane_model)

        object = object.select_by_index(inliers, invert=True)

        if(len(np.asarray(object.points)) < 10):
            break

#--------------------------------------------- Create meshes for each plane (visualizing) --------------------------------------
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

#--------------------------------------------- Find neighbor planes --------------------------------------
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

# print(planes_connectivity)
print(len(planes))
# print('objects planes normals=',objects_planes_normals)
# o3d.visualization.draw_geometries(np.asarray(planes))
# o3d.visualization.draw_geometries([mesh])

# print(plane_model)

#--------------------------------------------- Find corner edges --------------------------------------
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
    intersection_point.append(list())
    for triple in object:

        plane1 = objects_planes_normals[idx][triple[0]]
        plane2 = objects_planes_normals[idx][triple[1]]
        plane3 = objects_planes_normals[idx][triple[2]]

        planes_matrix = np.asarray([plane1, plane2, plane3])
        inter_point = np.linalg.solve(planes_matrix[:,0:3],planes_matrix[:,3])

        intersection_point[idx].append(inter_point)
        # print(point2)

#--------------------------------------------- Find lines between point --------------------------------------
objects_lines = list()
# for idx, object_points in enumerate(intersection_point):
#     lines = list()
#     for i in range(len(object_points)):
#         for j in range(len(object_points)):
#             points_vector = np.subtract(object_points[j],object_points[i])
#             for plane in objects_planes_normals[idx]:
#                 dot_result = abs(np.dot(points_vector,plane[0:3]))
#
#                 # Check if line is internal
#                 if (dot_result <= 0.001):
#                     lines.append((i,j))
#     objects_lines.append(lines)

all_objects_planes_points = list()
for idx, object_points in enumerate(intersection_point):
    object_planes_points = list()

    for object_normal in objects_planes_normals[idx]:
        points_on_plane = list()

        for point in object_points:
            point_vect = np.append(point,1)

            if(np.dot(np.asarray(point_vect),object_normal) <= 0.01):
                points_on_plane.append(point)

        object_planes_points.append(points_on_plane)

    all_objects_planes_points.append(object_planes_points)

pairs_points = list()

final_lineset = list()
final_objects_lines = list()

for object_planes_points in all_objects_planes_points:

    pairs_points = removeLines.remove_surface_lines(len(object_planes_points), object_planes_points)
    lineset_obj = list()

    for plane_points in pairs_points:
        iter = 0
        points_plane = list()
        lines_plane = list()
        for idx, pair in enumerate(plane_points):
            points_plane.append(pair[0])
            points_plane.append(pair[1])
            indx1 = iter
            iter = iter + 1
            indx2 = iter
            lines_plane.append([indx1, indx2])

        lineset_obj.append(o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(points_plane),
                lines=o3d.utility.Vector2iVector(lines_plane),
            ))

    final_objects_lines.append(np.sum(np.asarray(lineset_obj)))

objectsFinalPoints = list()
for object_points in intersection_point:
    intersection_point1 = np.asarray(object_points)
    finalObject = o3d.geometry.PointCloud()
    finalObject.points = o3d.utility.Vector3dVector(intersection_point1[:, :3])
    finalObject.colors = o3d.utility.Vector3dVector(intersection_point1[:, 0:3] / 255)
    finalObject.normals = o3d.utility.Vector3dVector(intersection_point1[:, 0:3])
    objectsFinalPoints.append(finalObject)

# o3d.visualization.draw_geometries(np.asarray(objectsFinalPoints))
o3d.visualization.draw_geometries([np.sum(final_objects_lines)])

# o3d.visualization.draw_geometries(np.asarray(final_meshes))


# o3d.visualization.draw_geometries(np.asarray(meshes))
# bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,o3d.utility.DoubleVector([radius, radius * 1.5]))
# bpa_mesh.compute_triangle_normals()
# o3d.visualization.draw_geometries([mesh2])
# o3d.io.write_triangle_mesh(output_path+"bpa_mesh11.ply", total_mesh,write_vertex_colors=False,write_vertex_normals=False,write_triangle_uvs=False,write_ascii=True)

print("Let's draw a box using o3d.geometry.LineSet.")
colors = [[1, 0, 0] for i in range(len(objects_lines[0]))]
lines_sets = list()
for idx, object_inter_points in enumerate(intersection_point):
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(object_inter_points),
        lines=o3d.utility.Vector2iVector(objects_lines[idx]),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    lines_sets.append(line_set)

o3d.visualization.draw_geometries(np.asarray(lines_sets))

