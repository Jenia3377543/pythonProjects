import open3d as o3d
import numpy as np
from sklearn.cluster import KMeans, OPTICS, cluster_optics_dbscan

def lod_mesh_export(mesh, lods, extension, path):
    mesh_lods={}
    for i in lods:
        # mesh_lod = mesh.simplify_quadric_decimation(i)
        mesh_lod = mesh.simplify_quadric_decimation(i)
        o3d.io.write_triangle_mesh(path+"lod_"+str(i)+extension, mesh_lod)
        mesh_lods[i]=mesh_lod
    print("generation of "+str(i)+" LoD successful")
    return mesh_lods

input_path="C:/Technion/semester6/project/pythonProjects/"
output_path="C:/Technion/semester6/project/pythonProjects/outputs"
dataname="arc.off"
datanameply="block_result.ply"

mesh = o3d.io.read_triangle_mesh(input_path+datanameply)
point_cloud_sampled = mesh.sample_points_poisson_disk(4400)
# print(np.asarray(point_cloud_sampled.points))
p1 = np.random.rand(3, 1)

# print(p1)
p1 = np.asarray([[-411.92, -399.1959, 42.260]])
# print(p1)



point_cloud = np.loadtxt(input_path+dataname,skiprows=2, max_rows=13631)
#
point_variance = 1
noise = np.random.normal(point_cloud,point_variance,point_cloud.shape)




# point_cloud = point_cloud + noise

# labels = np.array(point_cloud_sampled.cluster_dbscan(eps=2, min_points=25))
# print(labels)
# print(np.unique(labels))

# clust = OPTICS(min_samples=50, xi=.05, min_cluster_size=.05)
#
# # Run the fit
# clust.fit(np.asarray(point_cloud_sampled.points))
#
# labels_050 = cluster_optics_dbscan(reachability=clust.reachability_,
#                                    core_distances=clust.core_distances_,
#                                    ordering=clust.ordering_, eps=2)
# print(labels_050)

# kmeans = KMeans(n_clusters=2, random_state=0, max_iter=10000).fit(np.asarray(point_cloud_sampled.points))
# print(kmeans.cluster_centers_)
#
# pointsprediction = kmeans.predict(np.asarray(point_cloud_sampled.points))
#
#
# p3 = np.concatenate((point_cloud_sampled.points,np.asarray(kmeans.cluster_centers_)), axis=0)



point_cloud_sampled.colors = o3d.utility.Vector3dVector(point_cloud[:,0:3]/255)

distances = point_cloud_sampled.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)
radius = 3 * avg_dist

point_cloud_sampled.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=10))



min_distance = np.min(distances)
labels = np.asarray(point_cloud_sampled.cluster_dbscan(eps=6*min_distance, min_points=50))


print(np.unique(labels))
print(min_distance)

indexes1 = (labels==0)
indexes2 = (labels==1)

points = np.asarray(point_cloud_sampled.points)
normals = np.asarray(point_cloud_sampled.normals)
print(normals)

normals1 = normals[indexes1]
normals2 = normals[indexes2]

center1 = np.mean(points[indexes1],axis=0)
center2 = np.mean(points[indexes2],axis=0)

print(center1)
print(center2)

building1 = points[indexes1]
building2 = points[indexes2]


building1_from_center_to_points = np.subtract(building1,center1)
building1_from_center_to_points = building1_from_center_to_points/np.linalg.norm(building1_from_center_to_points,axis=1)[:,None]
# building1_from_center_to_points = building1_from_center_to_points

building2_from_center_to_points = np.subtract(building2,center2)
building2_from_center_to_points = building2_from_center_to_points/np.linalg.norm(building2_from_center_to_points,axis=1)[:,None]
# building2_from_center_to_points = building2_from_center_to_points

print("building1_from_center_to_points: ")
print(building1_from_center_to_points.shape)
print("building2_from_center_to_points: ")
print(building2_from_center_to_points.shape)

print(normals1.shape)
print(normals2.shape)
print(building1_from_center_to_points.shape)
print(building2_from_center_to_points.shape)

dot1 = np.array([np.dot(normals1[i], building1_from_center_to_points[i]) for i in range(len(normals1))])
dot2 = np.array([np.dot(normals2[i], building2_from_center_to_points[i]) for i in range(len(normals2))])

# dot1 = np.einsum('ij,ij->i', np.dot(normals1, building1_from_center_to_points), normals1)
# dot2 = np.einsum('ij,ij->i', np.dot(normals2, building2_from_center_to_points), normals2)

# dot1 = np.diag(np.dot(normals1,building1_from_center_to_points))
# dot2 = np.diag(np.dot(normals2,building2_from_center_to_points))

print("dot1: ")
print(dot1)
print("dot2: ")
print(dot2)

angle1 = np.arccos(dot1)
angle2 = np.arccos(dot2)

print("angle1: ")
print(angle1)
print("angle2: ")
print(angle2)

threshold = 3.14/2

normals1[angle1<threshold] = -normals1[angle1<threshold]
normals2[angle2<threshold] = -normals2[angle2<threshold]

normalsNew = np.concatenate((normals1,normals2),axis=0)
print(normalsNew.shape)
# point_cloud_sampled.normals = o3d.utility.Vector3dVector(normalsNew)

# p3 = np.concatenate((point_cloud_sampled.points,np.asarray([center1,center2])), axis=0)
# point_cloud_sampled.points = o3d.utility.Vector3dVector(p3)
o3d.visualization.draw_geometries([point_cloud_sampled])

bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(point_cloud_sampled,o3d.utility.DoubleVector([radius, radius * 1.5]))
o3d.visualization.draw_geometries([bpa_mesh])

# print(indexes1)
# print(indexes2)

dec_mesh = bpa_mesh.simplify_quadric_decimation(100000)

dec_mesh.remove_degenerate_triangles()
dec_mesh.remove_duplicated_triangles()
dec_mesh.remove_duplicated_vertices()
dec_mesh.remove_non_manifold_edges()

o3d.io.write_triangle_mesh(output_path+"bpa_mesh11.off", dec_mesh,write_vertex_colors=False,write_vertex_normals=False,write_triangle_uvs=False,write_ascii=True)


# my_lods = lod_mesh_export(bpa_mesh, [10000000], ".off", output_path)
#
#
# poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.1, linear_fit=False)[0]
# bbox = pcd.get_axis_aligned_bounding_box()
# p_mesh_crop = poisson_mesh.crop(bbox)
# o3d.io.write_triangle_mesh(output_path+"poisson_mesh.off", p_mesh_crop,write_vertex_colors=False,write_vertex_normals=False,write_triangle_uvs=False)

