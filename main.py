import pptk
import numpy as np
import open3d as o3d
# import arcpy

# P = np.random.rand(100,3)
# input_path="C:/Technion/semester6/project/pythonProjects/block.off"
# # df = np.genfromtxt(input_path, delimiter=' ', max_rows=13631)
# df = np.genfromtxt(input_path, delimiter=' ', max_rows=23794)
#
#
# v = pptk.viewer(df)

def lod_mesh_export(mesh, lods, extension, path):
    mesh_lods={}
    for i in lods:
        mesh_lod = mesh.simplify_quadric_decimation(i)
        o3d.io.write_triangle_mesh(path+"lod_"+str(i)+extension, mesh_lod)
        mesh_lods[i]=mesh_lod
    print("generation of "+str(i)+" LoD successful")
    return mesh_lods

input_path="C:/Technion/semester6/project/pythonProjects/"
output_path="C:/Technion/semester6/project/pythonProjects/outputs"
dataname="arc.off"


point_cloud = np.loadtxt(input_path+dataname,skiprows=2, max_rows=13631)

#
point_variance = 5
noise = np.random.normal(point_cloud,point_variance,point_cloud.shape)

# point_cloud = point_cloud + noise

with open('noised_arc.txt','w') as f:
    for row in range(np.size(point_cloud,0)):
         np.savetxt(f, [point_cloud[row,:]], fmt='%.2f')

#

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud[:,:3])
pcd.colors = o3d.utility.Vector3dVector(point_cloud[:,0:3]/255)
pcd.normals = o3d.utility.Vector3dVector(point_cloud[:,0:3])
o3d.visualization.draw_geometries([pcd])

# poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.1, linear_fit=False)[0]
#
#
# bbox = pcd.get_axis_aligned_bounding_box()
# p_mesh_crop = poisson_mesh.crop(bbox)



# o3d.io.write_triangle_mesh(output_path+"p_mesh_c.ply", p_mesh_crop)
# my_lods = lod_mesh_export(p_mesh_crop, [100000,50000,10000,1000,100], ".ply", output_path)
#
# o3d.visualization.draw_geometries([my_lods[100]])
