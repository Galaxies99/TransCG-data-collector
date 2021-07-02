import os
import numpy as np
from model import loadmodel


def pose_corrector(
    data_dir = 'data/scene1',
    id = 0,
    object_file_name_list = 'object_file_name_list.txt',
    perspective_num = 240,
	include_top = False,
	save_pose = False
):
	OBJECT_FILE_NAME_LIST_FILE_NAME=object_file_name_list
	object_file_name_list_file = open(OBJECT_FILE_NAME_LIST_FILE_NAME,'r')
	lines = object_file_name_list_file.readlines()
	objectfilenamelist = []

	for line in lines:
		if not (line == '\n'):
			objectfilenamelist.append(line.replace('\n','').replace('\r',''))	

	T_camera_aruco = np.load(os.path.join('configs', 'robot_calibration', '{}.npy'.format(id)))
	standard_pose_dir = os.path.join(data_dir, '0', 'pose')
	standard_model_list = []

	for filename in os.listdir(standard_pose_dir):
		obj_id, ext = os.path.splitext(filename)
		if ext != '.npy':
			continue
		try:
			obj_id = int(obj_id)
		except Exception:
			continue
		if obj_id < 0 or obj_id >= len(objectfilenamelist):
			continue
		standard_model_list.append(obj_id)

	res_model_list = []
	res_T = []

	if id == 0:
		res_model_list = standard_model_list
		for obj_id in standard_model_list:
			T_camera_object = np.load(os.path.join(standard_pose_dir, '{}.npy'.format(obj_id)))
			res_T.append(T_camera_object)
	else:
		for obj_id in standard_model_list:
			T_camera_object = []
			for i in range(0 if include_top else 1, perspective_num):
				base_pose_dir = os.path.join(data_dir, str(i), 'pose')
				if '{}.npy'.format(obj_id) not in os.listdir(base_pose_dir):
					continue
				T_camera_base_aruco = np.load(os.path.join('configs', 'robot_calibration', '{}.npy'.format(i)))
				T_camera_base_object = np.load(os.path.join(base_pose_dir, '{}.npy'.format(obj_id)))
				T_camera_object.append(T_camera_aruco.dot(np.linalg.inv(T_camera_base_aruco)).dot(T_camera_base_object))
			if T_camera_object != []:
				T_camera_object = np.stack(T_camera_object)
				T_camera_object = T_camera_object.mean(axis = 0)
				res_model_list.append(obj_id)
				res_T.append(T_camera_object)

	assert(len(res_model_list) == len(res_T))

	if save_pose:
		corrected_pose_dir = os.path.join(data_dir, str(id), 'corrected_pose')
		if os.path.exists(corrected_pose_dir) == False:
			os.makedirs(corrected_pose_dir)
		for i, obj_id in enumerate(res_model_list):
			T_camera_object = res_T[i]
			np.save(os.path.join(corrected_pose_dir, '{}.npy'.format(obj_id)), T_camera_object)

	return res_model_list, res_T
