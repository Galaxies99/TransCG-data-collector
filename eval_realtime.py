import os
import cv2
import math
import copy
import argparse
import numpy as np
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
parser.add_argument('--camera',default='realsense',help='realsense or kinect')
parser.add_argument('--id', default=0, help='The object ID', type=int)
parser.add_argument('--object_file_name_list',default='object_file_name_list.txt',help='ascii text file name that specifies the filenames of all possible objects')
parser.add_argument('--ip', default='10.52.25.177', help='IP address of the computer with Windows system', type=str)
parser.add_argument('--port', default=23333, help='The port that are used in netrelay', type=int)
FLAGS = parser.parse_args()

RES_DIR = FLAGS.res_dir
IP = FLAGS.ip
PORT = FLAGS.port

if FLAGS.camera == 'realsense':
	camera = RealSenseCamera()
elif FLAGS.camera == 'kinect':
	pass
else:
	raise ValueError('Invalid input for argument "camera"\n"camera" should be realsense or kinect')


MODEL_DIR=FLAGS.model_dir

OBJECT_FILE_NAME_LIST_FILE_NAME=FLAGS.object_file_name_list
OBJ_ID = FLAGS.id

# global variables
moving_speed = 5
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
	global runningflag, x, y, z, alpha, beta, gamma, transparency
	global moving_speed
	
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
	
	objectidlist=[OBJ_ID]
	obj_name, _ = os.path.splitext(objectfilenamelist[OBJ_ID])
    
	print('log:loading camera parameters')
	if FLAGS.camera == 'realsense':
		cam = np.array([927.17, 0.0, 651.32, 0.0, 927.37, 349.62, 0.0, 0.0, 1.0]).reshape((3, 3))
	elif FLAGS.camera == 'kinect':
		cam = np.array([631.54864502,0.0,638.43517329,0.,631.20751953,366.49904066,0.0,0.0,1.0]).reshape((3, 3))
	else:
		raise ValueError('Wrong type of camera')
	print('log:camera parameter:\n'+str(cam))
	
	# object pose vector format
	# [id, x, y, z, alpha, beta, gamma] angle in unit of degree
	posevector = get_pose_vector(OBJ_ID, 0, objectfilenamelist)
	models = None
	models_ply = None

	moving_speed = 5

	image, image_depth = img_from_cam()

	[_, x, y, z, alpha, beta, gamma] = posevector
	print('log:loading model '+os.path.join(MODEL_DIR, objectfilenamelist[OBJ_ID]))
	if models is None:
		assert models_ply is None
		models = loadmodel(MODEL_DIR, objectfilenamelist[OBJ_ID])
		downsample_name = objectfilenamelist[OBJ_ID].split('.ply')[0]+'_downsample.ply'
		if os.path.exists(os.path.join(MODEL_DIR,downsample_name)):
			models_ply = o3d.io.read_point_cloud(os.path.join(MODEL_DIR, downsample_name))
		else:
			models_ply = o3d.io.read_point_cloud(os.path.join(MODEL_DIR, objectfilenamelist[OBJ_ID]))
			models_ply.voxel_down_sample(DOWNSAMPLE_VOXEL_SIZE_M)
		print('log:model loaded')
	else:
		print('using cached model')
    
	T_tracker_camera = np.load('configs/T_tracker_camera.npy')
	transformation_file = 'results/transformation/{}.npy'.format(OBJ_ID)
	if not os.path.exists(transformation_file):
		raise ValueError('Transformation file not found')
	else:
		T_marker_object = np.load(transformation_file)
	
	s, id = client.start((IP, PORT))
	cmd_tracker = 'GetTracker'
		
	runningflag = True
	listener = keyboard.Listener(on_press=on_press,on_release=on_release)
	listener.start()
		
	while runningflag:
		notfound = False
		image, image_depth = img_from_cam()
		tracker_res = client.exec_cmd(s, cmd_tracker)
		tracker_js = formatter_str(tracker_res)

		try:
			T_tracker_marker = find_obj(tracker_js, obj_name)
		except ValueError:
			notfound = True

		if (not notfound) and T_marker_object is not None:
			T_camera_object = (np.linalg.inv(T_tracker_camera).dot(T_tracker_marker)).dot(T_marker_object)
			x, y, z, alpha, beta, gamma = get_pose(T_camera_object)

		pose = get_mat(x, y, z, alpha, beta, gamma)

		if notfound:
			rendered_image = image
		else:
			rendered_image = draw_model(image, pose, cam, models)
			rendered_image = (rendered_image * transparency + image * (1 - transparency)).astype(np.uint8)
			rendered_image = cv2.putText(rendered_image, 'x:%.3f y:%.3f z:%.3f alpha:%d beta:%d gamma:%d moving speed:%d' % (x, y, z, alpha, beta, gamma, moving_speed),\
				(20, image.shape[0] - 10),font, font_size, font_color, font_thickness)

		global state

		if state == 'confirm':
			rendered_image = cv2.putText(rendered_image, 'Confirm finishing this object?',(20, 25),font, font_size, font_color, font_thickness)
			rendered_image = cv2.putText(rendered_image, 'Press "enter" to confirm, others to resume',(20, 50),font, font_size, font_color, font_thickness)

		cv2.imshow('Annotater',rendered_image)
		cv2.waitKey(5)

	client.close(s)

if __name__ == '__main__':
	main()
