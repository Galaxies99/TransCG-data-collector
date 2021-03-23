import os
import json
import argparse
import numpy as np
from trans3d import get_mat
from xmlhandler import xmlReader

parser = argparse.ArgumentParser()
parser.add_argument('--id', default=0, help='Object ID', type=int)
parser.add_argument('--time', default=0, help='Times of sampling', type=int)
parser.add_argument('--object_file_name_list',default='object_file_name_list.txt',help='ascii text file name that specifies the filenames of all possible objects')
FLAGS = parser.parse_args()
ID = FLAGS.id
TIME = FLAGS.time
OBJECT_FILE_NAME_LIST_FILE_NAME=FLAGS.object_file_name_list

objectfilenamelist = []
with open(OBJECT_FILE_NAME_LIST_FILE_NAME, 'r') as f:
	lines = f.readlines()
	for line in lines:
		if not (line == '\n'):
			objectfilenamelist.append(line.replace('\n', '').replace('\r', ''))

filename = '{}-{}'.format(ID, TIME)
obj_name, _ = os.path.splitext(objectfilenamelist[ID])

with open('results/{}.json'.format(filename), 'r') as f:
    js = json.load(f)

js = js['TrackerData']['TargetPoses']

obj_found = False
for sub_js in js:
    if sub_js['TargetPose']['name'] == obj_name:
        T_tracker_marker = np.array(sub_js['TargetPose']['TransformationMatrix']).reshape(4, 4)
        obj_found = True

if not obj_found:
    raise ValueError('Object not found in {}.json!'.format(filename))

rd = xmlReader('results/{}.xml'.format(filename))
[[_, x, y, z, alpha, beta, gamma]] = rd.getposevectorlist()
T_camera_object = get_mat(x, y, z, alpha, beta, gamma)

print('Caclulating the transformation matrix ...')
try:
    T = T_camera_object.dot(np.linalg.inv(T_tracker_marker))
except Exception:
    print(e.values)
    exit()


print('The transformation matrix is: ', T)
print('Save the transformation matrix to file results/transformation/{}.npy'.format(ID))

with open('results/transformation/{}.npy'.format(ID), 'wb') as f:
    np.save(f, np.array(T))

