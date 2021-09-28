import open3d as o3d
import numpy as np

eps = 6
min_point_for_plane = 50
distance_threshold = 0.1
roof_color = [255, 0, 0]
wall_color = [0, 0, 0]

def segmentObjects(pcd):
    distances = pcd.compute_nearest_neighbor_distance()
    min_distance = np.min(distances)
    labels = np.asarray(pcd.cluster_dbscan(eps=eps * min_distance,
                                                       min_points=min_point_for_plane))
    objects_labels = np.unique(labels)

    # Check if clustering found some objects, depends on params we choose in clustering step
    if (len(objects_labels) == 1 & -1 in objects_labels):
        print("Error in segmentation!")
        return
    points = np.asarray(pcd.points)
    #  Start getting objects
    objects = []
    for label in np.unique(labels):
        object = o3d.geometry.PointCloud()
        object.points = o3d.utility.Vector3dVector(points[labels == label, :3])
        object.colors = o3d.utility.Vector3dVector(points[labels == label, 0:3] / 255)
        object.normals = o3d.utility.Vector3dVector(points[labels == label, 0:3])
        objects.append(object)

    print("objects count : {objects}".format(objects=len(np.unique(labels))))
    return objects

def fitPlanesForObjects(objects):
    if(len(objects) <= 0):
      print("There no objects to fit!")
      return
    all_object_planes = []
#     Start to fit planes for objects we detected in segmentObject(self) step
    for idx, object in enumerate(objects):
      object_planes = []

      hasPlanes = False
      while (True):

        # Find plane with RANSAC
        plane_model, inliers = object.segment_plane(distance_threshold=distance_threshold, ransac_n=4, num_iterations=1000)
        # Select points that belong to the plane
        inlier_cloud = object.select_by_index(inliers)
        if (len(np.asarray(inlier_cloud.points)) > 30):
          rand_colors = np.abs(np.random.randn(3, 1))
          inlier_cloud.paint_uniform_color(rand_colors)
          # Append plane to planes list of the building
          object_planes.append((inlier_cloud, plane_model))
          hasPlanes = True

        # Select points that don't belong to any plane
        object = object.select_by_index(inliers, invert=True)

        if (len(np.asarray(object.points)) < 10):
          break

      # If there are any planes in building -> append building to building list
      if(hasPlanes):
        all_object_planes.append(object_planes)
    print("There found {count} objects".format(count=len(all_object_planes)))
    return all_object_planes

def pick_points(pcd, coordinate_frame):
    print("")
    print(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) Afther picking points, press q for close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.add_geometry(coordinate_frame, reset_bounding_box=False)
    vis.run()  # user picks points
    roof_or_wall = input("Enter roof or wall (R or W)")
    plane_color = [0, 0, 0]
    selected_pcd_color = [0, 0, 0]

    if(roof_or_wall == 'R'):
        plane_color = roof_color
    else:
        plane_color = wall_color

    colors_titles = ['Red', 'Green', 'Blue']
    for idx, color_title in enumerate(colors_titles):
        selected_pcd_color[idx] = int(input(f"Enter selected pcd color {color_title} value"))
    selected_points_idx = vis.get_picked_points()
    selected_points = pcd.select_by_index(selected_points_idx)
    left_points = pcd.select_by_index(selected_points_idx, invert=True)
    print(f"selected points {selected_points}")
    selected_points.colors = o3d.utility.Vector3dVector(np.multiply(np.ones(np.shape(np.asarray(selected_points.points))),np.asarray(selected_pcd_color)))
    left_points.colors = o3d.utility.Vector3dVector(np.multiply(np.ones(np.shape(np.asarray(left_points.points))),np.asarray(plane_color)))
    vis.destroy_window()
    print("")
    return (selected_points+left_points)

input_path="./"
datanameply=["block_result.ply","buildings.obj","buildings2.obj","arc_result.ply","block.off"]

mesh = o3d.io.read_triangle_mesh(input_path+datanameply[0])
pcd = mesh.sample_points_poisson_disk(8000)
pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.points)/255)
pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.points))
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=pcd.get_center())

# pick_points(point_cloud_sampled)
# colors = np.multiply(np.ones(np.asarray(point_cloud_sampled.points).shape),255)

# objects = segmentObjects(pcd)
# all_objects_planes = fitPlanesForObjects(objects)
# new_objects_with_color = []
# for idx, objects_planes in enumerate(all_objects_planes):
#     print(f"building #{idx}")
#     new_object = []
#     for plane_pcd, plane_model in objects_planes:
#         print(f"current plane model: {plane_model}")
#         new_object.append(pick_points(plane_pcd, coordinate_frame))
#     new_objects_with_color.append(np.sum(np.asarray(new_object),axis=0))
# all_new_pcd = np.sum(np.asarray(new_objects_with_color),axis=0)
# o3d.visualization.draw_geometries([all_new_pcd])
#
# np.savetxt('sampled_pcd2',np.concatenate((all_new_pcd.points,all_new_pcd.colors),axis=1), fmt='%.5f')
np.savetxt('sampled_pcd2',np.concatenate((pcd.points,pcd.colors),axis=1), fmt='%.5f')

