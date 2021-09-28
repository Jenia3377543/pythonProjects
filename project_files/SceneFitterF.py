import open3d as o3d
import numpy as np
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
from kneed import KneeLocator

from .PlaneF import Plane
from .BuildingF import Building

class SceneFitter:
  eps = 6
  min_point_for_plane = 30
  distance_threshold = 0.1

  def __init__(self, pointCloud):
    self.pointCloud = pointCloud
    self.buildings = list()
    self.objects = list()
    self.meshes = list()
    self.distances = self.pointCloud.compute_nearest_neighbor_distance()
    self.min_distance = np.min(self.distances)

  def get_building_triangle_mesh(self):
    meshes = []
    for building in self.buildings:
      vert, faces, building_uv_maps, building_uv_idx, materials = building.get_building_triangles()
      temp_mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vert),
                                o3d.utility.Vector3iVector(faces))


      temp_mesh.compute_vertex_normals()
      temp_mesh.triangle_uvs = building_uv_maps
      temp_mesh.textures = materials

      temp_mesh.triangle_material_ids = building_uv_idx

      meshes.append(temp_mesh)

    return np.sum(np.asarray(meshes))
  '''
   Suppose for now that only building can be in point cloud (there no: cars, and so on)
  '''
  def segmentObjects(self):
    # counts, bins = np.histogram(self.distances)
    # plt.hist(bins[:-1], bins, weights=counts)
    # plt.ylabel("Points count")
    # plt.xlabel("Nearest distance")
    # plt.show()

    minPts = 6
    neighbors = NearestNeighbors(n_neighbors=minPts)
    neighbors_fit = neighbors.fit(self.pointCloud.points)
    distances, indices = neighbors_fit.kneighbors(self.pointCloud.points)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]

    kn = KneeLocator(range(len(distances)), distances, curve='convex', direction='increasing')

    plt.plot(distances, label=r"$\bar{\epsilon}$ - distance to nearest 6 point")
    plt.plot(kn.knee,distances[kn.knee], "ro")
    plt.legend()

    plt.title(r"Optimal $\bar{\epsilon}$ values")
    plt.ylabel(r"$\bar{\epsilon}$")
    plt.xlabel("Points index")
    plt.show()

    self.min_distance = 3*distances[kn.knee]
    self.min_point_for_plane = 3*minPts
    labels = np.asarray(self.pointCloud.cluster_dbscan(eps=self.min_distance,
                                                       min_points=self.min_point_for_plane))

    print(f"knee={kn.knee}")
    # labels = np.asarray(self.pointCloud.cluster_dbscan(eps=3*distances[kn.knee],
    #                                                    min_points=3*minPts))
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
      building = Building(object)
      building.threshold_neighbor_planes = self.min_distance
      hasPlanes = False
      while (True):

        # Find plane with RANSAC
        plane_model, inliers = object.segment_plane(distance_threshold=self.distance_threshold, ransac_n=4, num_iterations=1000)
        # Select points that belong to the plane
        inlier_cloud = object.select_by_index(inliers)
        if (len(np.asarray(inlier_cloud.points)) > self.min_point_for_plane):
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

    return np.sum(np.asarray(self.meshes))

  def meshes_with_planes_scene(self):
    if(len(self.buildings) <= 0):
      print("There is no building to find mesh!")
    meshes_with_planes = list()
    pcd_with_planes = list()
    for building in self.buildings:
      mesh,pcd = building.building_mesh_with_planes()
      meshes_with_planes.append(mesh)
      pcd_with_planes.append(pcd)

    return np.sum(np.asarray(meshes_with_planes)), np.sum(np.asarray(pcd_with_planes))

  def find_intersections_points_and_lines(self):
    lines = list()
    for building in self.buildings:
      building.find_neighbor_planes()
      building.find_intersections()
      lines.append(building.find_lines())

    return np.sum(np.asarray(lines))

  def get_total_inter_points(self):
    pcd = o3d.geometry.PointCloud()
    for building in self.buildings:
      pcd += building.get_inter_points()
    return pcd

  def get_planes_centers_of_mass(self):
    centers_of_mass = []

    pc_centers_of_mass = o3d.geometry.PointCloud()


    for building in self.buildings:
      for building_plane in building.planes:
        centers_of_mass.append(building_plane.center_of_mass)

    centers_of_mass = np.asarray(centers_of_mass)

    pc_centers_of_mass.points = o3d.utility.Vector3dVector(centers_of_mass[:, :3])
    pc_centers_of_mass.colors = o3d.utility.Vector3dVector(centers_of_mass[:, 0:3] / 255)
    pc_centers_of_mass.normals = o3d.utility.Vector3dVector(centers_of_mass[:, 0:3])

    return pc_centers_of_mass
