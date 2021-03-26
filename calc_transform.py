import os
import json
import argparse
import numpy as np
from trans3d import get_mat
from xmlhandler import xmlReader
from jsonhandler import find_obj

parser = argparse.ArgumentParser()
parser.add_argument('--id', default=0, help='Object ID', type=int)
parser.add_argument('--time', default=0, help='Times of sampling', type=int)
FLAGS = parser.parse_args()
ID = FLAGS.id
TIME = FLAGS.time
filename = '{}-{}'.format(ID, TIME)

T_tracker_marker = np.load('results/{}.npy'.format(filename))

rd = xmlReader('results/{}.xml'.format(filename))
[[_, x, y, z, alpha, beta, gamma]] = rd.getposevectorlist()
T_camera_object = get_mat(x, y, z, alpha, beta, gamma)

T_tracker_camera = np.load('configs/T_tracker_camera.npy')

print('Caclulating the transformation matrix ...')
try:
    T_marker_object = (np.linalg.inv(T_tracker_marker).dot(T_tracker_camera)).dot(T_camera_object)
except Exception:
    print(e.values)
    exit()

print('The transformation matrix is: ', T_marker_object)
print('Save the transformation matrix to file results/transformation/{}.npy'.format(ID))

with open('results/transformation/{}.npy'.format(ID), 'wb') as f:
    np.save(f, np.array(T_marker_object))

