import open3d as o3d
import trimesh
import numpy as np

input_path="C:/Technion/semester6/project/pythonProjects/"
output_path="C:/Technion/semester6/project/pythonProjects/outputs"
dataname="arc.off"


point_cloud = np.loadtxt(input_path+dataname,skiprows=2, max_rows=13631)

#
point_variance = 1
noise = np.random.normal(point_cloud,point_variance,point_cloud.shape)

# point_cloud = point_cloud + noise

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud[:,:3])
pcd.colors = o3d.utility.Vector3dVector(point_cloud[:,0:3]/255)
# pcd.normals = o3d.utility.Vector3dVector(point_cloud[:,0:3])

# pcd = o3d.io.read_point_cloud("pointcloud.ply")
pcd.estimate_normals()

# estimate radius for rolling ball
distances = pcd.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)
radius = 1.5 * avg_dist

mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
           pcd,
           o3d.utility.DoubleVector([radius, radius * 2]))

# create the triangular mesh with the vertices and faces from open3d
tri_mesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
                          vertex_normals=np.asarray(mesh.vertex_normals))

trimesh.convex.is_convex(tri_mesh)
file_obj = open('noised_arc.off','w')
trimesh.exchange.export.export_mesh(tri_mesh, file_obj, file_type='off', resolver=None)
file_obj.close()