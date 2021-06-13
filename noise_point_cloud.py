#!/usr/bin/python

import numpy as np
import sys
import open3d as o3d


input_path="C:/Technion/semester6/project/pythonProjects/arc.off"
# input_path=sys.argv[1:]
output_path="C:/Technion/semester6/project/pythonProjects/outputs"

# file = open(str(input_path),'r')
file = open(input_path,'r')
rows = file.readlines()

file_data = rows[1].split(sep=' ')

points_count = file_data[0]
triangles_count = file_data[1]
param3 = file_data[2]

points = np.loadtxt(input_path, skiprows=2, max_rows=int(points_count))
#
point_variance = 1
noise = np.random.normal(points,point_variance,points.shape)

points_noised = points + noise




# Noise points
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_noised[:,:3])
pcd.colors = o3d.utility.Vector3dVector(points_noised[:,0:3]/255)

distances = pcd.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)
radius = 3 * avg_dist

# pcd.normals = o3d.utility.Vector3dVector(point_cloud[:,0:3])
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=10))

o3d.visualization.draw_geometries([pcd])
# Create mesh


bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,o3d.utility.DoubleVector([radius, radius * 2]))
o3d.visualization.draw_geometries([bpa_mesh])

# poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.1, linear_fit=False)[0]
# bbox = pcd.get_axis_aligned_bounding_box()
# p_mesh_crop = poisson_mesh.crop(bbox)
# o3d.visualization.draw_geometries([poisson_mesh])

with open(input_path + ".noised_data",'w') as f:
    f.write(rows[0])
    f.write(rows[1])

    for row in range(np.size(points_noised,0)):
         np.savetxt(f, [points_noised[row,:]], fmt='%.2f')

    for row in range(int(triangles_count)):
         f.write(rows[row +int(points_count) + 2])
f.close()
