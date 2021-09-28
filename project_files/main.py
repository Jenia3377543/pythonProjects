import open3d as o3d
import numpy as np
from project_files.SceneFitterF import SceneFitter

POINTS_TO_SAMPLE = 5000
export_mesh = False

input_path="./input/"
output_path="./output/"

dataname=["block_result.ply","point_cloud_file"]
file_to_load = dataname[1]
pcd = []

if(file_to_load.endswith('.ply')):
    mesh = o3d.io.read_triangle_mesh(input_path+dataname[0])
    pcd = mesh.sample_points_poisson_disk(POINTS_TO_SAMPLE)
else:
    point_cloud = np.loadtxt(input_path + dataname[1])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:,0:3])
    pcd.colors = o3d.utility.Vector3dVector(point_cloud[:,3:]/255)
    pcd.normals = o3d.utility.Vector3dVector(point_cloud[:,0:3])

o3d.visualization.draw_geometries([pcd])


sceneFitter = SceneFitter(pcd)

sceneFitter.segmentObjects()
sceneFitter.fitPlanesForObjects()

o3d.visualization.draw_geometries([np.sum(np.asarray(sceneFitter.meshes_scene()))])
o3d.visualization.draw_geometries([np.sum(sceneFitter.find_intersections_points_and_lines())])

mesh, pcd = sceneFitter.meshes_with_planes_scene()

o3d.visualization.draw_geometries([mesh, pcd])

pcd = sceneFitter.get_total_inter_points()

o3d.visualization.draw_geometries([pcd])

o3d.visualization.draw_geometries([sceneFitter.get_building_triangle_mesh()])

if(export_mesh):
    o3d.io.write_triangle_mesh(output_path+"textured_buildings.obj", sceneFitter.get_building_triangle_mesh(), write_triangle_uvs=True)

