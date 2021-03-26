import os
import cv2
import math
import copy
import argparse
import numpy as np
import open3d as o3d
from pynput import keyboard
from renderer import Renderer,draw_model
from camera.camera import RealSenseCamera
from model import Model3D, loadmodel, cachemodel
from transforms3d.euler import quat2euler, euler2quat
from transforms3d.quaternions import mat2quat, quat2mat
from xmlhandler import xmlWriter, xmlReader, get_pose_vector
from trans3d import get_mat, pos_quat_to_pose_4x4, get_pose, pose_4x4_rotation


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='models', help='ply model files directory path')
parser.add_argument('--save_xml',default='True',help='True for saving xml, False for not saving xml, when checking the result, this arg should be set as False')
parser.add_argument('--xml_dir',default='results',help='output xml file directory')
parser.add_argument('--camera',default='realsense',help='realsense or kinect')
parser.add_argument('--id', default=0, help='The object ID', type=int)
parser.add_argument('--object_file_name_list',default='object_file_name_list.txt',help='ascii text file name that specifies the filenames of all possible objects')
parser.add_argument('--time',default=0,help='time', type=int)
FLAGS = parser.parse_args()

if FLAGS.save_xml == 'True':
	IS_SAVE_XML = True
	XML_DIR = FLAGS.xml_dir
elif FLAGS.save_xml == 'False':
	IS_SAVE_XML = False
else:
	raise ValueError('Invalid input for argument "save_xml"\n"save_xml" should be True or False')

if FLAGS.camera == 'realsense':
	camera = RealSenseCamera()
elif FLAGS.camera == 'kinect':
	pass
else:
	raise ValueError('Invalid input for argument "camera"\n"camera" should be realsense or kinect')


MODEL_DIR=FLAGS.model_dir
TIME = FLAGS.time

OBJECT_FILE_NAME_LIST_FILE_NAME=FLAGS.object_file_name_list
OBJ_ID = FLAGS.id

# global variables
moving_speed = 5
x, y, z = 0.0, 0.0, 0.0
alpha, beta, gamma = 0.0, 0.0, 0.0
runningflag = True
state = 'normal'
DOWNSAMPLE_VOXEL_SIZE_M = 0.005

def on_press(key):
	global x,y,z,alpha,beta,gamma,runningflag,moving_speed
	global state
	if state == 'normal':
		try:		
			if key.char=='d':
				x += 0.001 * moving_speed
			elif key.char == 'a':
				x -= 0.001 * moving_speed
			elif key.char == 'w':
				y -= 0.001 * moving_speed
			elif key.char == 's':
				y += 0.001 * moving_speed
			elif key.char == 'e':
				z += 0.001 * moving_speed
			elif key.char == 'c':
				z -= 0.001 * moving_speed
			elif key.char == 'j':
				angle = 1.0 * moving_speed
				x, y, z, alpha, beta, gamma = get_pose(pose_4x4_rotation(get_mat(x, y, z, alpha, beta, gamma), angle, axis='X'))
			elif key.char == 'l':
				angle = -1.0 * moving_speed
				x, y, z, alpha, beta, gamma = get_pose(pose_4x4_rotation(get_mat(x, y, z, alpha, beta, gamma), angle, axis='X'))
			elif key.char == 'i':
				angle = 1.0 * moving_speed
				x, y, z, alpha, beta, gamma = get_pose(pose_4x4_rotation(get_mat(x, y, z, alpha, beta, gamma), angle, axis='Y'))
			elif key.char == 'k':
				angle = -1.0 * moving_speed
				x, y, z, alpha, beta, gamma = get_pose(pose_4x4_rotation(get_mat(x, y, z, alpha, beta, gamma), angle, axis='Y'))
			elif key.char == 'u':
				angle = 1.0 * moving_speed
				x, y, z, alpha, beta, gamma = get_pose(pose_4x4_rotation(get_mat(x, y, z, alpha, beta, gamma), angle, axis='Z'))
			elif key.char == 'm':
				angle = -1.0 * moving_speed
				x, y, z, alpha, beta, gamma = get_pose(pose_4x4_rotation(get_mat(x, y, z, alpha, beta, gamma), angle, axis='Z'))
			elif key.char == ']':
				moving_speed *= 5
			elif key.char == '[':
				if moving_speed > 1:
					moving_speed /= 5
		except AttributeError:
			if key == keyboard.Key.enter:
				state = 'confirm'
	elif state == 'confirm':
		if key == keyboard.Key.enter:
			runningflag = False
			return False
		else:
			state = 'normal'

def on_release(key):
	pass


def img_from_cam():
	image, image_depth = camera.get_rgbd()
	image = (image * 255).astype(np.uint8)
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
	return image, image_depth


def main():
	global runningflag, x, y, z, alpha, beta, gamma
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
	posevector = get_pose_vector(OBJ_ID, TIME, objectfilenamelist)
	models = None
	models_ply = None

	while True:
		moving_speed = 5

		image, image_depth = img_from_cam()

		textimage = copy.deepcopy(image)
		textimage = cv2.putText(textimage, 'Input an intager to select ply file, 0 for exiting', (10, 30), font, font_size, font_color, font_thickness)
		textimage = cv2.putText(textimage, '0: exit',(10, 55), font, font_size, font_color, font_thickness)
		textimage = cv2.putText(textimage, '1: ' + objectfilenamelist[OBJ_ID],(10, 80),font, font_size, font_color, font_thickness)
		cv2.imshow('Annotater',textimage)
		getkey = cv2.waitKey(1)

		if getkey < ord('0') or getkey > ord('1'):
			continue
			
		if getkey == ord('0'):
			break
		
		print(posevector)
		[_, x, y, z, alpha, beta, gamma] = posevector
		textimage2 = copy.deepcopy(image)
		textimage2 = cv2.putText(textimage2, 'loading model '+os.path.join(MODEL_DIR, objectfilenamelist[OBJ_ID]), (10, 30), font, font_size, font_color, font_thickness)
		cv2.imshow('Annotater',textimage2)
		cv2.waitKey(5)
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
		
		runningflag = True
		listener = keyboard.Listener(on_press=on_press,on_release=on_release)
		listener.start()
		
		while runningflag:
			image, image_depth = img_from_cam()
			pose = get_mat(x,y,z, alpha, beta, gamma)
			rendered_image=draw_model(image, pose, cam, models)
			rendered_image = cv2.putText(rendered_image, 'x:%.3f y:%.3f z:%.3f alpha:%d beta:%d gamma:%d moving speed:%d' % (x,y,z,alpha,beta,gamma,moving_speed),\
			(20, image.shape[0] - 10),font, font_size, font_color, font_thickness)
			global state

			if state == 'confirm':
				rendered_image = cv2.putText(rendered_image, 'Confirm finishing this object?',(20, 25),font, font_size, font_color, font_thickness)
				rendered_image = cv2.putText(rendered_image, 'Press "enter" to confirm, others to resume',(20, 50),font, font_size, font_color, font_thickness)
			cv2.imshow('Annotater',rendered_image)
			cv2.waitKey(5)

		posevector = [OBJ_ID, x, y, z, alpha, beta, gamma]

		if IS_SAVE_XML:
			if not os.path.exists(XML_DIR):
				print('log: create directory '+XML_DIR)
				os.mkdir(XML_DIR)
			mainxmlWriter = xmlWriter()
			mainxmlWriter.objectlistfromposevectorlist(posevectorlist=[posevector], objdir=MODEL_DIR, objnamelist=objectfilenamelist, objidlist=objectidlist)
			mainxmlWriter.writexml(xmlfilename=os.path.join(XML_DIR, f'{OBJ_ID}-{TIME}.xml'))

if __name__ == '__main__':
	main()
