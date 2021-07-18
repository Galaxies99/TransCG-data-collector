import os
import argparse
import numpy as np
from cv2 import cv2
from pynput import keyboard
from renderer import draw_model
from model import loadmodel
from pose_corrector import PoseCorrector


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default = 'models', help = 'ply model files directory path', type = str)
parser.add_argument('--data_dir', default = 'data', help = 'data for visualization', type = str)
parser.add_argument('--id', default = 0, help = 'the perspective ID', type = int)
parser.add_argument('--object_file_name_list', default = 'object_file_name_list.txt', help = 'ascii text file name that specifies the filenames of all possible objects', type = str)
parser.add_argument('--corrected', action = 'store_true', help = 'whether to use the corrected poses.')
FLAGS = parser.parse_args()

MODEL_DIR=FLAGS.model_dir
OBJECT_FILE_NAME_LIST_FILE_NAME=FLAGS.object_file_name_list
id = int(FLAGS.id)
PRE_DATA_DIR = FLAGS.data_dir
DATA_DIR = os.path.join(PRE_DATA_DIR, str(FLAGS.id))
CORRECTED = FLAGS.corrected

# global variables
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
	
	cam = np.array([927.17, 0.0, 651.32, 0.0, 927.37, 349.62, 0.0, 0.0, 1.0]).reshape((3, 3))

	obj_id_list = range(len(objectfilenamelist))

	models = []
	for obj_id in obj_id_list:
		models.append(None)

	if CORRECTED:
		corrector = PoseCorrector(object_file_name_list = OBJECT_FILE_NAME_LIST_FILE_NAME, perspective_num = 240)
		res_model_list, res_T = corrector.correct_pose(PRE_DATA_DIR, id, include_top = False, save_pose = False)
		image = cv2.imread(os.path.join(DATA_DIR, 'rgb1.png'))
		rendered_image = image
		for i, obj_id in enumerate(res_model_list):
			T_camera_object = res_T[i]
			obj_filename = objectfilenamelist[obj_id]
			if models[obj_id] is None:
				print('log: loading model', obj_filename)
				models[obj_id] = loadmodel(MODEL_DIR, obj_filename)
			rendered_image = draw_model(rendered_image, T_camera_object, cam, models[obj_id])
	else:
		image = cv2.imread(os.path.join(DATA_DIR, 'rgb1.png'))
		rendered_image = image
		pose_dir = os.path.join(DATA_DIR, 'pose')

		for filename in os.listdir(pose_dir):
			obj_id, ext = os.path.splitext(filename)
			if ext != '.npy':
				continue
			try:
				obj_id = int(obj_id)
			except Exception:
				continue
			if obj_id < 0 or obj_id >= len(objectfilenamelist):
				continue
			obj_filename = objectfilenamelist[obj_id]
			if models[obj_id] is None:
				print('log: loading model', obj_filename)
				models[obj_id] = loadmodel(MODEL_DIR, obj_filename)
			T_camera_object = np.load(os.path.join(pose_dir, '{}.npy'.format(obj_id)))
			rendered_image = draw_model(rendered_image, T_camera_object, cam, models[obj_id])


	runningflag = True
	listener = keyboard.Listener(on_press=on_press,on_release=on_release)
	listener.start()
	
	while runningflag:
		final = (rendered_image * transparency + image * (1 - transparency)).astype(np.uint8)
		final = cv2.putText(final, 'Transparency: %.1f' % transparency, (20, final.shape[0] - 10),font, font_size, font_color, font_thickness)
		cv2.imshow('Evaluator', final)
		cv2.waitKey(5)


if __name__ == '__main__':
	main()
