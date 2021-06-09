#!/usr/bin/python

import numpy as np
import sys


input_path="C:/Technion/semester6/project/pythonProjects/"
# input_path=sys.argv[1:]
output_path="C:/Technion/semester6/project/pythonProjects/outputs"
dataname="arc.off"

file = open(str(input_path+dataname),'r')
rows = file.readlines()

file_data = rows[1].split(sep=' ')

points_count = file_data[0]
triangles_count = file_data[1]
param3 = file_data[2]

points = np.loadtxt(input_path, skiprows=2, max_rows=int(points_count))
#
point_variance = 1
noise = np.random.normal(points,point_variance,points.shape)
print(noise)
# points_noised = points + noise
#
#
# with open(input_path + ".noised_data",'w') as f:
#     f.write(rows[0])
#     f.write(rows[1])
#
#     for row in range(np.size(points_noised,0)):
#          np.savetxt(f, [points_noised[row,:]], fmt='%.2f')
#
#     for row in range(int(triangles_count)):
#          f.write(rows[row +int(points_count) + 2])
# f.close()