import os
import math
import copy
import argparse
import numpy as np
from cv2 import cv2
import open3d as o3d
from pynput import keyboard
import netrelay.client as client
from renderer import Renderer,draw_model
from camera.camera import RealSenseCamera
from jsonhandler import formatter_str, find_obj
from model import Model3D, loadmodel, cachemodel
from transforms3d.euler import quat2euler, euler2quat
from transforms3d.quaternions import mat2quat, quat2mat
from xmlhandler import xmlWriter, xmlReader, get_pose_vector
from trans3d import get_mat, pos_quat_to_pose_4x4, get_pose, pose_4x4_rotation


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='models', help='ply model files directory path')
parser.add_argument('--res_dir',default='results',help='output file directory')
parser.add_argument('--camera',default='realsense_D435',help='realsense_D435 or realsense_L515')
parser.add_argument('--object_file_name_list',default='object_file_name_list.txt',help='ascii text file name that specifies the filenames of all possible objects')
parser.add_argument('--id', default=-1, help='The object ID, -1 for all objects', type=int)
parser.add_argument('--ip', default='10.52.25.177', help='IP address of the computer with Windows system', type=str)
parser.add_argument('--port', default=23333, help='The port that are used in netrelay', type=int)
FLAGS = parser.parse_args()

RES_DIR = FLAGS.res_dir
IP = FLAGS.ip
PORT = FLAGS.port
ID = FLAGS.id

if FLAGS.camera == 'realsense_D435':
	camera = RealSenseCamera(type='D435')
elif FLAGS.camera == 'realsense_L515':
	camera = RealSenseCamera(type='L515')
else:
	raise ValueError('Invalid input for argument "camera"\n"camera" should be realsense_D435 or realsense_L515.')

MODEL_DIR=FLAGS.model_dir

OBJECT_FILE_NAME_LIST_FILE_NAME=FLAGS.object_file_name_list

# global variables
x, y, z = 0.0, 0.0, 0.0
alpha, beta, gamma = 0.0, 0.0, 0.0
runningflag = True
state = 'normal'
DOWNSAMPLE_VOXEL_SIZE_M = 0.005
transparency = 0.5

def on_press(key):
	global transparency, runningflag
	global state
	if state == 'normal':
		try:		
			if key.char == '.':
				if transparency <= 0.9:
					transparency += 0.1
			elif key.char == ',':
				if transparency >= 0.1:
					transparency -= 0.1
			elif key.char == 'q':
				state = 'quit'
		except AttributeError:
			pass
	if state == 'quit':
		runningflag = False
		return False


def on_release(key):
	pass


def img_from_cam():
	image, image_depth = camera.get_rgbd()
	image = (image * 255).astype(np.uint8)
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
	return image, image_depth


def main():
	global runningflag, transparency
	
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

	obj_id_list = []
	if ID != -1:
		obj_id_list.append(ID)
	else:
		obj_id_list = range(len(objectfilenamelist))
    
	print('log:loading camera parameters')
	if 'realsense' in FLAGS.camera:
		cam = np.array([927.17, 0.0, 651.32, 0.0, 927.37, 349.62, 0.0, 0.0, 1.0]).reshape((3, 3))
	elif 'kinect' in FLAGS.camera:
		cam = np.array([631.54864502,0.0,638.43517329,0.,631.20751953,366.49904066,0.0,0.0,1.0]).reshape((3, 3))
	else:
		raise ValueError('Wrong type of camera')
	print('log:camera parameter:\n'+str(cam))
	
	models = []

	if models == []:
		for obj_id in obj_id_list:
			models.append(None)
	else:
		print('using cached model')
    
	T_tracker_camera = np.load('configs/T_tracker_camera.npy')

	s, _ = client.start((IP, PORT))
	cmd_tracker = 'GetTracker'
		
	runningflag = True
	listener = keyboard.Listener(on_press=on_press,on_release=on_release)
	listener.start()
		
	while runningflag:
		image, _ = img_from_cam()
		tracker_res = client.exec_cmd(s, cmd_tracker)
		try:
			tracker_js = formatter_str(tracker_res)
		except ValueError:
			continue
		rendered_image = image
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

		rendered_image = (rendered_image * transparency + image * (1 - transparency)).astype(np.uint8)
		rendered_image = cv2.putText(rendered_image, 'Transparency: %.1f' % transparency, (20, image.shape[0] - 10),font, font_size, font_color, font_thickness)
		cv2.imshow('Evaluator', rendered_image)
		cv2.waitKey(5)

	client.close(s)

if __name__ == '__main__':
	main()
