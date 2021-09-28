import open3d as o3d
import numpy as np
import cv2
import TextureGenerator as tg
import PlaneClassifier as classifier



class Building:
  threshold_neighbor_planes = 5

  def __init__(self, pointCloud):
    self.planes = list()
    self.lines = list()
    self.meshes = list()
    self.near_planes_connectivity = list()
    self.planes_connectivity = list()
    self.intersections = list()
    self.lines = list()
    self.line_set = list()
    self.pointCloud = pointCloud
    self.center_of_mass = pointCloud.get_center()

    cm = o3d.geometry.PointCloud()
    cm.points = o3d.utility.Vector3dVector([pointCloud.get_center()])

    self.bounding_radius = 2*np.max(pointCloud.compute_point_cloud_distance(cm))
    print("bounding radius = {radius}".format(radius=self.bounding_radius))




  def get_building_triangles(self):
    all_intersections = []
    for plane in self.planes:
      # if(len(plane.inter_points)>4):
      #   continue
      for point in plane.inter_points:
        all_intersections.append(point)

    all_intersections = np.asarray(all_intersections)

    # all_intersections = np.unique(np.concatenate(all_intersections,axis=0),axis=1)
    # all_intersections = np.unique(all_intersections,axis=1)

    print(f'all intersections: {all_intersections}')

    building_triangles = []
    building_materials = []
    building_uv_maps = []
    building_uv_idx = []


    materials = [
      ("./textures/wall_tail1.png",(2.4,1.5)),
      ("./textures/wall_tail1.png",(2.4,1.5)),
      ("./textures/roof_tile2.png",(0.5,0.8)),
    ]


    materials = [tg.TextureTail(cv2.cvtColor(cv2.imread(materials[idx][0]), cv2.COLOR_BGR2RGB),materials[idx][1]) for idx in range(len(materials))]

    for idx, plane in enumerate(self.planes):
      # if (len(plane.inter_points) > 4):
      #   continue
      planes_triangles, uv_map = plane.get_plane_triangles(self.center_of_mass)

      for item in uv_map:
        building_uv_maps.append(item)

      for item in [idx] * len(planes_triangles):
        building_uv_idx.append(item)


      building_materials.append(
        # o3d.geometry.Image(plane.texture)

        o3d.geometry.Image(tg.tailed_texture(np.asarray(plane.inter_points),
          materials[classifier.classifyPlane(plane.normal[0:3])]
          # cv2.cvtColor(cv2.imread(materials[idx]), cv2.COLOR_BGR2RGB)
        ))
        # if idx in [1,2] else
        # o3d.geometry.Image(
        #   cv2.cvtColor(cv2.imread(materials[idx]), cv2.COLOR_BGR2RGB)
        # )
      )
      for triangle in planes_triangles:
        building_triangles.append(triangle)

    print(f"building_uv_maps: {building_uv_maps}")
    building_uv_maps = o3d.utility.Vector2dVector(np.asarray(building_uv_maps))
    building_uv_idx = o3d.utility.IntVector(np.asarray(building_uv_idx))
    return all_intersections, np.asarray([[np.where(np.all(np.abs(np.subtract(all_intersections,point))<1, axis=1))[0][0] for point in triangle] for triangle in building_triangles]), building_uv_maps, building_uv_idx, building_materials
    # return all_intersections, np.asarray(faces)



  def get_inter_points(self):
    pcd = o3d.geometry.PointCloud()
    for plane in self.planes:
      pcd+=plane.get_inter_points_as_pcd()
    return pcd

  def is_inside(self, point):
    dist = np.linalg.norm(np.subtract(point, self.center_of_mass))
    if(dist <= self.bounding_radius):
      return True
    return False

  def building_mesh(self):
    if(len(self.planes) <= 0):
      print("There is no planes to find meshes!")
      return

    for plane in self.planes:
      self.meshes.append(plane.get_mesh())

    return np.sum(np.asarray(self.meshes))

  def building_mesh_with_planes(self):
    if(len(self.planes) <= 0):
      print("There is no planes to find meshes!")
      return
    mesh_with_planes = list()
    pcd_with_planes = list()
    for plane in self.planes:
      mesh,pcd = plane.find_plane()
      mesh_with_planes.append(mesh)
      pcd_with_planes.append(pcd)

    return np.sum(np.asarray(mesh_with_planes)), np.sum(np.asarray(pcd_with_planes))

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
        inter_point = np.linalg.solve(planes_matrix[:, 0:3], -planes_matrix[:, 3])
        if(self.is_inside(inter_point)):
          self.intersections.append(inter_point)

          self.planes[triple[0]].inter_points.append(inter_point)
          self.planes[triple[1]].inter_points.append(inter_point)
          self.planes[triple[2]].inter_points.append(inter_point)

    for plane in self.planes:
      plane = np.unique(plane)

  def find_lines(self):

    line_sets = []
    for plane in self.planes:
      index = 0
      final_lines = []
      intersections = []
      plane_points_pairs = plane.remove_surface_lines()

      for pair in plane_points_pairs:
        intersections.append(pair[0])
        intersections.append(pair[1])
        final_lines.append((index, index+1))
        index = index + 2
      line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(intersections),
        lines=o3d.utility.Vector2iVector(final_lines),
      )

      rand_color = [0, 0, 1]
      colors = [rand_color for i in range(len(final_lines))]
      line_set.colors = o3d.utility.Vector3dVector(colors)
      line_sets.append(line_set)

    return np.sum(line_sets)
