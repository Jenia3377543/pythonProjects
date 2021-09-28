import open3d as o3d
import numpy as np
import itertools
import cv2
import TextureGenerator as tg



class Plane:
  def __init__(self, points, normal):
    txtG = tg.TextureGenerator(points)
    self.points = points
    self.center_of_mass = []
    self.inter_points = list()
    self.normal = normal
    self.mesh = None
    self.distances = points.compute_nearest_neighbor_distance()
    self.avg_dist = np.mean(self.distances)
    self.texture = txtG.solid_texture()

  def get_plane_triangles(self, object_center_of_mass):
    points = np.asarray(self.inter_points)
    normal = self.normal[0:3]

    if(np.dot(normal,object_center_of_mass)+self.normal[3]>0):
      normal = -normal

    center_of_mass = np.mean(points, axis=0)
    possible_axis = np.subtract(points, points[0])
    p = possible_axis[np.argsort(np.linalg.norm(possible_axis, axis=1))[-1]]
    q = np.cross(normal, p)

    indexes = np.argsort(
      np.arctan2(
        np.dot(np.cross(np.subtract(points, center_of_mass), q), normal),
        np.dot(np.cross(np.subtract(points, center_of_mass), p), normal)
      )
    )

    ordered_points = points[indexes]
    initial_point = ordered_points[0]
    points_count = len(ordered_points)

    triangles_list = list()
    start_idx = 1
    while (start_idx + 1 < points_count):
      triangles_list.append([initial_point,
                             ordered_points[start_idx],
                             ordered_points[start_idx + 1]])
      start_idx = start_idx + 1

    temp_uv_map = [
      [
      [0, 1],
      [0, 0],
      [1, 0]
    ],
      [
      [0, 1],
      [1, 0],
      [1, 1]
     ]
    ]


    uv_map = []
    for idx in range(len(triangles_list)):
      for point in temp_uv_map[idx%2]:
        uv_map.append(point)

    return np.asarray(triangles_list), np.asarray(uv_map)

  def get_inter_points_as_pcd(self):
    points = np.asarray(self.inter_points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(points / 255)
    pcd.normals = o3d.utility.Vector3dVector(points)
    return pcd

  def get_mesh(self):
    radius = 1.5 * self.avg_dist
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
      self.points,
      o3d.utility.DoubleVector([radius, radius * 2]))
    mesh.compute_vertex_normals()
    return mesh

  def find_plane(self):
    print("Inter plane points: {count}".format(count=len(self.inter_points)))
    print("Inter plane points: {points}".format(points=self.inter_points))
    vectors = np.subtract(np.asarray(self.inter_points)[1:,:],np.asarray(self.inter_points[0]))
    indexes = np.argsort(np.linalg.norm(vectors,axis=1))

    vectorA = vectors[indexes[0]]
    vectorB = vectors[indexes[1]]

    normA = np.linalg.norm(vectorA)
    normB = np.linalg.norm(vectorB)


    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray([self.inter_points[0],self.inter_points[0]]))
    pcd.colors = o3d.utility.Vector3dVector(np.asarray([self.inter_points[0],self.inter_points[0]]) / 255)
    pcd.normals = o3d.utility.Vector3dVector(np.asarray([vectorA, vectorB]))

    S = np.asarray([
      [normA, 0, 0, 0],
      [0, normB, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
    ])
    X = vectorA / normA
    Y = vectorB / normB
    Z = np.cross(X, Y)

    R = [
      [X[0], Y[0], Z[0], 0],
      [X[1], Y[1], Z[1], 0],
      [X[2], Y[2], Z[2], 0],
      [0, 0, 0, 1],
    ]

    O = np.mean(self.inter_points, axis=0)

    print("Translation: {translation}".format(translation=O))


    T = [
      [1, 0, 0, O[0]],
      [0, 1, 0, O[1]],
      [0, 0, 1, O[2]],
      [0, 0, 0, 1],
    ]

    # M = np.matmul(np.matmul(T,R),S)
    M = np.matmul(T, np.matmul(R, S))

    mesh_box = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=0.001)
    mesh_box.compute_vertex_normals()
    mesh_box.translate(-mesh_box.get_center()).transform(M)
    mesh_box.paint_uniform_color([0.9, 0.1, 0.1])

    return mesh_box,pcd

  #input: intesection_points array.
  # intersection_points[index]=a list of intersection points which are on plane number index
  # output: relevant_intersec_points list.
  # relevant_intersec_points[index]=a list of intersection points which are on plane number index, and are relevant
  def remove_surface_lines(self):
      points = self.inter_points
      if (len(points) <= 0):
        print("There is no intersection points in current plane!")
        return

      # center_of_mass = []
      # for axis in range(3):
      #   min_value = np.min(np.asarray(points)[:,axis])
      #   max_value = np.max(np.asarray(points)[:,axis])
      #   center_of_mass.append((max_value-min_value)/2+min_value)
      self.center_of_mass = np.mean(points,axis=0)

      print("center_of_mass={center_of_mass}".format(center_of_mass = self.center_of_mass))
      all_pairs = list(itertools.combinations(points,2))
      num_of_points = len(points)
      dists_pairs = list()

      for pair in all_pairs:
          # finding the vertical distance from the center of the mass to the line between the 2 points in pair:
          dist = np.linalg.norm(np.cross(np.subtract(pair[1],pair[0]), np.subtract(self.center_of_mass,pair[1])))/np.linalg.norm(np.subtract(pair[1],pair[0]))
          dist_pair=(dist, pair)
          dists_pairs.append(dist_pair)


      dists_array=np.asarray(dists_pairs)
      #for plane with N intersaction points, we will choose the N lines
      # which have the biggest distance from center of mass:

      indexes = np.argsort(dists_array[:,0])[-num_of_points:]

      return dists_array[indexes,1]

  def remove_surface_lines2(self):
      points = self.inter_points
      center_of_mass=np.mean(points,axis=0)
      self.center_of_mass = center_of_mass

      point_pairs = []

      vectors_to_center_of_mass = []
      index=1
      for point in points:
        vector=(index,np.subtract(center_of_mass,point))
        vectors_to_center_of_mass.append(vector)
        index=index+1

      for vector in vectors_to_center_of_mass:
        min_angle = 360
        min_vector = vector[1]
        for vector2 in vectors_to_center_of_mass:
          if(vector[0]>=vector2[0]):
            continue
      #       check angle
          vector = np.asarray(vector[1])
          vector2 = np.asarray(vector2[1])
          arcos = np.dot(vector, vector2)/(np.linalg.norm(vector), np.linalg.norm(vector2))

          print("arcos={arcos}".format(arcos=arcos))

          temp_angle = np.arccos(arcos)
          if(min_angle>temp_angle):
            min_vector = vector2
            min_angle = temp_angle

        point_pairs.append((
          np.subtract(center_of_mass, vector),
          np.subtract(center_of_mass, min_vector)
        ))

      return point_pairs
