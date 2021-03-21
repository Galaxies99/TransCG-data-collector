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
from frame_transform import frameTransformer
from model import Model3D,loadmodel,cachemodel
from trans3d import get_mat,pos_quat_to_pose_4x4
from transforms3d.euler import quat2euler, euler2quat
from transforms3d.quaternions import mat2quat, quat2mat
from xmlhandler import xmlWriter,xmlReader,getposevectorlist


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='models', help='ply model files directory path')
parser.add_argument('--save_xml',default='True',help='True for saving xml, False for not saving xml, when checking the result, this arg should be set as False')
parser.add_argument('--xml_dir',default='results',help='output xml file directory')
parser.add_argument('--camera',default='realsense',help='realsense or kinect')
parser.add_argument('--resume',default='True',help='xml file path to resume annotation')
parser.add_argument('--id', default=0, help='The experiment ID')
parser.add_argument('--object_file_name_list',default='object_file_name_list.txt',help='ascii text file name that specifies the filenames of all possible objects')
parser.add_argument('--time',default=0,help='time')
FLAGS = parser.parse_args()

if FLAGS.save_xml == 'True':
	IS_SAVE_XML = True
	XML_DIR = FLAGS.xml_dir
elif FLAGS.save_xml == 'False':
	IS_SAVE_XML = False
else:
	raise ValueError('Invalid input for argument "save_xml"\n"save_xml" should be True or False')

if FLAGS.resume == 'True':
	IS_RESUME = True
elif FLAGS.resume == 'False':
	IS_RESUME = False
else:
	raise ValueError('Invalid input for argument "resume"\n"resume" should be True or False')

if FLAGS.camera == 'realsense':
	camera = RealSenseCamera()
elif FLAGS.camera == 'kinect':
	pass
else:
	raise ValueError('Invalid input for argument "camera"\n"camera" should be realsense or kinect')


MODEL_DIR=FLAGS.model_dir
TIME = FLAGS.time

OBJECT_FILE_NAME_LIST_FILE_NAME=FLAGS.object_file_name_list
EXPER_ID = FLAGS.id

# global variables
moving_speed = 5
x, y, z = 0.0, 0.0, 0.0
alpha, beta, gamma = 0.0, 0.0, 0.0
runningflag = True
state = 'normal'
DOWNSAMPLE_VOXEL_SIZE_M = 0.005

# camera
camera = RealSenseCamera()

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
				alpha += 1.0 * moving_speed
			elif key.char == 'l':
				alpha -= 1.0 * moving_speed
			elif key.char == 'i':
				beta += 1.0 * moving_speed
			elif key.char == 'k':
				beta -= 1.0 * moving_speed
			elif key.char == 'u':
				gamma += 1.0 * moving_speed
			elif key.char == 'm':
				gamma -= 1.0 * moving_speed
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
	
	objectidlist=[EXPER_ID]
	print('log:loaded object id list:',end='')
	print(objectidlist)
    
	print('log:loading camera parameters')
	if FLAGS.camera == 'realsense':
		cam = np.array([927.17, 0.0, 651.32, 0.0, 927.37, 349.62, 0.0, 0.0, 1.0]).reshape((3, 3))
	elif FLAGS.camera == 'kinect':
		cam = np.array([631.54864502,0.0,638.43517329,0.,631.20751953,366.49904066,0.0,0.0,1.0]).reshape((3, 3))
	else:
		raise ValueError('Wrong type of camera')
	print('log:camera parameter:\n'+str(cam))
	
	# posevectorlist format
	# [object1 posevector,object2 posevector]
	# object pose vector format
	# [id,x,y,z,alpha,beta,gamma] angle in unit of degree
	# posevectorlist[objectid] = [id,x,y,z,alpha,beta,gamma]
	posevectorlist = getposevectorlist(objectidlist, IS_RESUME, EXPER_ID, XML_DIR)
	models = []
	models_ply = []
	for i in range(len(objectidlist)):
		models.append(None)
		models_ply.append(None)

	while True:
		moving_speed = 5
		existed_object_file_name_list = []
		for objectid in objectidlist:
			existed_object_file_name_list.append(objectfilenamelist[objectid])

		image, image_depth = img_from_cam()

		textimage = copy.deepcopy(image)
		textimage = cv2.putText(textimage, 'Input an intager to select ply file, 0 for exiting', (10, 30), font, font_size, font_color, font_thickness)
		textimage = cv2.putText(textimage, '0: exit',(10, 55), font, font_size, font_color, font_thickness)
		for i in range(len(existed_object_file_name_list)):
			if i < 9:
				textimage = cv2.putText(textimage, str(i+1)+': '+existed_object_file_name_list[i],(10, 55 + 25 * (i + 1)),font, font_size, font_color, font_thickness)
			else:
				textimage = cv2.putText(textimage, chr(ord('a') + i - 9)+': '+existed_object_file_name_list[i],(10, 55 + 25 * (i + 1)),font, font_size, font_color, font_thickness)
		cv2.imshow('Annotater',textimage)
		getkey = cv2.waitKey(1)
		no_key_flag = False
		if getkey < ord('0') or getkey > ord('z') or (getkey < ord('a') and getkey > ord('9')):
			no_key_flag = True
		if no_key_flag:
			continue
		if getkey - ord('0') <= 9:
			plyfilenumber = getkey - ord('0')
		else:
			plyfilenumber = getkey - ord('a') + 10
	
		if plyfilenumber == 0:
			break
		else:
			plyfilenumber -= 1
		# the plyfilenumber is out of range
		if plyfilenumber >= len(existed_object_file_name_list):
			continue
		[_,x,y,z,alpha,beta,gamma]=posevectorlist[plyfilenumber]
		textimage2 = copy.deepcopy(image)
		textimage2 = cv2.putText(textimage2, 'loading model '+os.path.join(MODEL_DIR,existed_object_file_name_list[plyfilenumber]),(10, 30),font, font_size, font_color, font_thickness)
		cv2.imshow('Annotater',textimage2)
		cv2.waitKey(5)
		print('log:loading model '+os.path.join(MODEL_DIR,existed_object_file_name_list[plyfilenumber]))
		if models[plyfilenumber] is None:
			assert models_ply[plyfilenumber] is None
			models[plyfilenumber] = loadmodel(MODEL_DIR,existed_object_file_name_list[plyfilenumber])
			downsample_name = existed_object_file_name_list[plyfilenumber].split('.ply')[0]+'_downsample.ply'
			if os.path.exists(os.path.join(MODEL_DIR,downsample_name)):
				models_ply[plyfilenumber] = o3d.io.read_point_cloud(os.path.join(MODEL_DIR, downsample_name))
			else:
				models_ply[plyfilenumber] = o3d.io.read_point_cloud(os.path.join(MODEL_DIR,existed_object_file_name_list[plyfilenumber]))
				models_ply[plyfilenumber].voxel_down_sample(DOWNSAMPLE_VOXEL_SIZE_M)
			print('log:model loaded')
		else:
			print('using cached model')
		
		runningflag = True
		listener = keyboard.Listener(on_press=on_press,on_release=on_release)
		listener.start()
		
		while runningflag:
			image, image_depth = img_from_cam()
			pose = get_mat(x,y,z, alpha, beta, gamma)
			rendered_image=draw_model(image, pose, cam, models[plyfilenumber])
			rendered_image = cv2.putText(rendered_image, 'x:%.3f y:%.3f z:%.3f alpha:%d beta:%d gamma:%d moving speed:%d' % (x,y,z,alpha,beta,gamma,moving_speed),\
			(20, image.shape[0] - 10),font, font_size, font_color, font_thickness)
			global state

			if state == 'confirm':
				rendered_image = cv2.putText(rendered_image, 'Confirm finishing this object?',(20, 25),font, font_size, font_color, font_thickness)
				rendered_image = cv2.putText(rendered_image, 'Press "enter" to confirm, others to resume',(20, 50),font, font_size, font_color, font_thickness)
			cv2.imshow('Annotater',rendered_image)
			cv2.waitKey(5)

		posevectorlist[plyfilenumber]=[objectidlist[plyfilenumber], x, y, z, alpha, beta, gamma]

		if IS_SAVE_XML:
			if not os.path.exists(XML_DIR):
				print('log: create directory '+XML_DIR)
				os.mkdir(XML_DIR)
			mainxmlWriter = xmlWriter()
			mainxmlWriter.objectlistfromposevectorlist(posevectorlist=posevectorlist, objdir=MODEL_DIR, objnamelist=objectfilenamelist, objidlist=objectidlist)
			mainxmlWriter.writexml(xmlfilename=os.path.join(XML_DIR, f'{EXPER_ID}-{TIME}.xml'))

if __name__ == '__main__':
	main()
