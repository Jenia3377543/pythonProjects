import numpy as np
import itertools
from collections import namedtuple


#input: intesection_points array.
# intersection_points[index]=a list of intersection points which are on plane number index
# output: relevant_intersec_points list.
# relevant_intersec_points[index]=a list of intersection points which are on plane number index, and are relevant
def remove_surface_lines(num_of_planes, intesection_points):
    relevant_intersec_points=list()
    for plane_index in range(num_of_planes):
            points=intesection_points[plane_index]
            center_of_mass=np.mean(points)
            all_pairs=list(itertools.combinations(points,2))
            num_of_points=len(points)
            dists_pairs=list()

            for pair in all_pairs:
                # finding the vertical distance from the center of the mass to the line between the 2 points in pair:
                dist=np.dot(np.subtract(pair[1],pair[0]), np.subtract(center_of_mass,pair[1]))/np.linalg.norm(np.subtract(pair[1],pair[0]))
                dist_pair=(dist, pair)
                dists_pairs.append(dist_pair)


            dists_array=np.asarray(dists_pairs)
            #for plane with N intersaction points, we will choose the N lines
            # which have the biggest distance from center of mass:
            indexes = np.argsort(dists_array[:,0])[-num_of_points:]

            relevant_intersec_points.append(dists_array[indexes,1])
    return relevant_intersec_points

# array = [
#     [[0,0,0],[10,0,0],[10,10,0],[0,10,0]],
#     [[0,0,0],[12,18,1312],[10,11,32],[65,45,0],[13,312,33],[46,23,7],[37,97,101]]
# ]
#
# result = remove_lines(2, array)
# print(result)
