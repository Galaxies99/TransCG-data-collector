import os
import argparse
import numpy as np
from cv2 import cv2
import netrelay.client as client
from renderer import draw_model
from camera.camera import RealSenseCamera
from jsonhandler import formatter_str, find_obj
from model import loadmodel


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='models', help='ply model files directory path')
parser.add_argument('--data_dir',default='data',help='data directory')
parser.add_argument('--object_file_name_list',default='object_file_name_list.txt',help='ascii text file name that specifies the filenames of all possible objects')
parser.add_argument('--id', default=-1, help='the scene ID')
parser.add_argument('--time', default=0, help='the perspective ID')
parser.add_argument('--ip', default='10.52.25.177', help='IP address of the computer with Windows system', type=str)
parser.add_argument('--port', default=23333, help='The port that are used in netrelay', type=int)
FLAGS = parser.parse_args()

DATA_DIR = FLAGS.data_dir
if os.path.exists(DATA_DIR) == False:
	os.makedirs(DATA_DIR)
ID = FLAGS.id
TIME = FLAGS.time
IP = FLAGS.ip
PORT = FLAGS.port
camera1 = RealSenseCamera(type='D435', use_infrared=True)
camera2 = RealSenseCamera(type='L515')
MODEL_DIR=FLAGS.model_dir
DATA_DIR = os.path.join(DATA_DIR, 'scene{}'.format(ID))
CUR_DATA_DIR = os.path.join(DATA_DIR, str(TIME))
if os.path.exists(CUR_DATA_DIR) == False:
	os.makedirs(CUR_DATA_DIR)

OBJECT_FILE_NAME_LIST_FILE_NAME = FLAGS.object_file_name_list

# global variables
x, y, z = 0.0, 0.0, 0.0
alpha, beta, gamma = 0.0, 0.0, 0.0
runningflag = True
state = 'normal'
DOWNSAMPLE_VOXEL_SIZE_M = 0.005
transparency = 0.5
image, image2, image_depth, image_depth2, image_infrared_left, image_infrared_right = None, None, None, None, None, None
pose = []


def main():
	global runningflag, transparency
	global image, image2, image_depth, image_depth2, image_infrared_left, image_infrared_right, pose
	
	font_size = 0.5
	font_thickness = 1
	font = cv2.FONT_HERSHEY_SIMPLEX
	font_color = (255,0,255)	

	object_file_name_list_file = open(OBJECT_FILE_NAME_LIST_FILE_NAME,'r')
	lines = object_file_name_list_file.readlines()
	objectfilenamelist = []
	for line in lines:
		if not (line == '\n'):
			objectfilenamelist.append(line.replace('\n','').replace('\r',''))
	print('log:loaded object file name list:',end='')
	print(objectfilenamelist)

	obj_id_list = range(len(objectfilenamelist))
    
	cam = np.array([927.17, 0.0, 651.32, 0.0, 927.37, 349.62, 0.0, 0.0, 1.0]).reshape((3, 3))
	
	models = []
	pose = [None] * len(obj_id_list)

	if models == []:
		for obj_id in obj_id_list:
			models.append(None)
	else:
		print('using cached model')
    
	T_tracker_camera = np.load('configs/T_tracker_camera.npy')

	s, _ = client.start((IP, PORT))
	cmd_tracker = 'GetTracker'
	
	image, image_depth, image_infrared_left, image_infrared_right = camera1.get_full_image()
	image2, image_depth2 = camera2.get_full_image()
	tracker_res = client.exec_cmd(s, cmd_tracker)
	tracker_js = formatter_str(tracker_res)

	for i, obj_id in enumerate(obj_id_list):
		notfound = False
		obj_filename = objectfilenamelist[obj_id]
		obj_name, _ = os.path.splitext(obj_filename)
		try:
			T_tracker_marker = find_obj(tracker_js, obj_name)
		except ValueError:
			notfound = True
			
		transformation_file = 'results/transformation/{}.npy'.format(obj_id)
		if not os.path.exists(transformation_file):
			notfound = True
		else:
			T_marker_object = np.load(transformation_file)
	
		if not notfound:
			T_camera_object = (np.linalg.inv(T_tracker_camera).dot(T_tracker_marker)).dot(T_marker_object)
			if models[i] is None:
				print('log: loading model', obj_filename)
				models[i] = loadmodel(MODEL_DIR, obj_filename)
			rendered_image = draw_model(rendered_image, T_camera_object, cam, models[i])
			pose[i] = T_camera_object
		else:
			pose[i] = None

		print('log: saving data')
		POSE_DIR = os.path.join(CUR_DATA_DIR, 'pose')
		if os.path.exists(POSE_DIR) == False:
			os.makedirs(POSE_DIR)
		cv2.imwrite(os.path.join(CUR_DATA_DIR, 'rgb1.png'), image)
		cv2.imwrite(os.path.join(CUR_DATA_DIR, 'rgb2.png'), image2)
		cv2.imwrite(os.path.join(CUR_DATA_DIR, 'depth1.png'), image_depth)
		cv2.imwrite(os.path.join(CUR_DATA_DIR, 'depth2.png'), image_depth2)
		cv2.imwrite(os.path.join(CUR_DATA_DIR, 'ir1-left.png'), image_infrared_left)
		cv2.imwrite(os.path.join(CUR_DATA_DIR, 'ir1-right.png'), image_infrared_right)
		for i, p in enumerate(pose):
			if p is None:
				continue
			else:
				np.save(os.path.join(POSE_DIR, '{}.npy'.format(i)), p)
		print('log: finish saving data')

	client.close(s)


if __name__ == '__main__':
	main()