import open3d as o3d
import numpy as np

class SceneFitter:
  eps = 6
  min_point_for_plane = 50

  def __init__(self, pointCloud):
    self.pointCloud = pointCloud
    self.buildings = list()
    self.objects = list()
    self.meshes = list()
    self.distances = self.pointCloud.compute_nearest_neighbor_distance()
    self.min_distance = np.min(self.distances)

  '''
   Suppose for now that only building can be in point cloud (there no: cars, and so on)
  '''
  def segmentObjects(self):
    labels = np.asarray(self.pointCloud.cluster_dbscan(eps=self.eps * self.min_distance,
                                                       min_points=self.min_point_for_plane))
    objects_labels = np.unique(labels)

    # Check if clustering found some objects, depends on params we choose in clustering step
    if(len(objects_labels) == 1 & -1 in objects_labels):
      print("Error in segmentation!")
      return
    points = np.asarray(self.pointCloud.points)
    #  Start getting objects
    for label in np.unique(labels):
      object = o3d.geometry.PointCloud()
      object.points = o3d.utility.Vector3dVector(points[labels == label, :3])
      object.colors = o3d.utility.Vector3dVector(points[labels == label, 0:3] / 255)
      object.normals = o3d.utility.Vector3dVector(points[labels == label, 0:3])
      self.objects.append(object)

    print("objects count : {objects}".format(objects=len(np.unique(labels))))

  def fitPlanesForObjects(self):
    if(len(self.objects) <= 0):
      print("There no objects to fit!")
      return

#     Start to fit planes for objects we detected in segmentObject(self) step
    for idx, object in enumerate(self.objects):
      building = Building()
      while (True):
        hasPlanes = False
        # Find plane with RANSAC
        plane_model, inliers = object.segment_plane(distance_threshold=0.01, ransac_n=4, num_iterations=1000)
        # Select points that belong to the plane
        inlier_cloud = object.select_by_index(inliers)
        if (len(np.asarray(inlier_cloud.points)) > 30):
          rand_colors = np.abs(np.random.randn(3, 1))
          inlier_cloud.paint_uniform_color(rand_colors)
          # Append plane to planes list of the building
          building.planes.append(Plane(inlier_cloud, plane_model))
          hasPlanes = True

        # Select points that don't belong to any plane
        object = object.select_by_index(inliers, invert=True)

        if (len(np.asarray(object.points)) < 10):
          break

      # If there are any planes in building -> append building to building list
      if(hasPlanes):
        self.buildings.append(building)
    print("There found {count} buildings".format(count=len(self.buildings)))

  def meshes_scene(self):
    if(len(self.buildings) <= 0):
      print("There is no building to find mesh!")

    for building in self.buildings:
      self.meshes.append(building.building_mesh())

    return self.meshes

  def find_intersections_points_and_lines(self):
    lines = list()
    for building in self.buildings:
      building.find_neighbor_planes()
      building.find_intersections()
      lines.append(building.find_lines())

    return np.asarray(lines)

class Building:
  threshold_neighbor_planes = 5

  def __init__(self):
    self.planes = list()
    self.lines = list()
    self.meshes = list()
    self.near_planes_connectivity = list()
    self.planes_connectivity = list()
    self.intersections = list()
    self.lines = list()
    self.line_set = list()

  def building_mesh(self):
    if(len(self.planes) <= 0):
      print("There is no planes to find meshes!")
      return

    for plane in self.planes:
      self.meshes.append(plane.get_mesh())

    return np.asarray(self.meshes)

  def find_neighbor_planes(self):
    # Find neighbor planes
    for ind1, plane1 in enumerate(self.planes):
      self.near_planes_connectivity.append(list())
      for ind2, plane2 in enumerate(self.planes):
        if (ind1 == ind2):
          continue

        distance = np.min(plane1.points.compute_point_cloud_distance(plane2.points))
        if (distance < self.threshold_neighbor_planes):
          self.near_planes_connectivity[ind1].append(ind2)

    # Find planes to cross in order to get point on edge
    planes_connectivity = list()
    for conn_index, plane_connectivity_list in enumerate(self.near_planes_connectivity):
      for connection in plane_connectivity_list:
        common_planes = np.intersect1d(plane_connectivity_list, self.near_planes_connectivity[connection])

        for common_item in common_planes:
          planes_connectivity.append([conn_index, connection, common_item])

    object_planes_points2_sorted = list()
    for triple in planes_connectivity:
        object_planes_points2_sorted.append(np.sort(triple))

    self.planes_connectivity = np.unique(object_planes_points2_sorted, axis=0)

  def find_intersections(self):

    for triple in self.planes_connectivity:
        plane1 = self.planes[triple[0]].normal
        plane2 = self.planes[triple[1]].normal
        plane3 = self.planes[triple[2]].normal

        planes_matrix = np.asarray([plane1, plane2, plane3])
        inter_point = np.linalg.solve(planes_matrix[:, 0:3], planes_matrix[:, 3])
        self.intersections.append(inter_point)

  def find_lines(self):
    for i in range(len(self.intersections)):
        for j in range(len(self.intersections)):
          points_vector = np.subtract(self.intersections[j], self.intersections[i])
          for plane in self.planes:
            dot_result = abs(np.dot(points_vector, plane.normal[0:3]))
            # print(dot_result)
            if (dot_result <= 0.001):
              print("dot result <= 0.001")
              self.lines.append((i, j))
    print("lines count: {count}".format(count=len(self.lines)))
    self.line_set = o3d.geometry.LineSet(
      points=o3d.utility.Vector3dVector(self.intersections),
      lines=o3d.utility.Vector2iVector(self.lines),
    )
    colors = [[1, 0, 0] for i in range(len(self.lines))]
    self.line_set.colors = o3d.utility.Vector3dVector(colors)

    return self.line_set

class Plane:
  def __init__(self, points, normal):
    self.points = points
    self.normal = normal
    self.mesh = None
    self.distances = points.compute_nearest_neighbor_distance()
    self.avg_dist = np.mean(self.distances)

  def get_mesh(self):
    radius = 1.5 * self.avg_dist
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
      self.points,
      o3d.utility.DoubleVector([radius, radius * 2]))
    mesh.compute_vertex_normals()
    return mesh

class Line:
  def __init__(self):
    self.points = list()


input_path="C:/Technion/semester6/project/pythonProjects/"
output_path="C:/Technion/semester6/project/pythonProjects/outputs"
datanameply="block_result.ply"
dataname="block.off"

mesh = o3d.io.read_triangle_mesh(input_path+datanameply)
point_cloud_sampled = mesh.sample_points_poisson_disk(5000)

sceneFitter = SceneFitter(point_cloud_sampled)

sceneFitter.segmentObjects()
sceneFitter.fitPlanesForObjects()

o3d.visualization.draw_geometries([np.sum(np.asarray(sceneFitter.meshes_scene()))])
o3d.visualization.draw_geometries([np.sum(sceneFitter.find_intersections_points_and_lines())])

