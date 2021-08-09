import os
import argparse
import numpy as np


class PoseCorrector(object):
	def __init__(self, **kwargs):
		super(PoseCorrector, self).__init__()
		object_file_name_list = kwargs.get('object_file_name_list', 'object_file_name_list.txt')
		with open(object_file_name_list, 'r') as object_filename_file:
			self.obj_filename_list = []
			for line in object_filename_file.readlines():
				if not (line == '\n'):
					self.obj_filename_list.append(line.replace('\n', '').replace('\r', ''))
		self.perspective_num = kwargs.get('perspective_num', 240)
		self.T_camera_aruco = [None] * self.perspective_num
		self.cam_calibration_path = kwargs.get('cam_calibration_path', os.path.join('configs', 'robot_calibration'))
		perspective_pair_weight_path = kwargs.get('perspective_pair_weight_path', None)
		if perspective_pair_weight_path is None:
			self.perspective_pair_weight = np.ones((self.perspective_num, self.perspective_num))
		else:
			try:
				self.perspective_pair_weight = np.load(perspective_pair_weight_path)
			except Exception:
				self.perspective_pair_weight = np.ones((self.perspective_num, self.perspective_num))

	def get_camera_aruco(self, cam_id):
		if cam_id < 0 or cam_id >= self.perspective_num:
			raise AttributeError('camera perspective ID out of range')
		if self.T_camera_aruco[cam_id] is None:
			self.T_camera_aruco[cam_id] = np.load(os.path.join(self.cam_calibration_path, '{}.npy'.format(cam_id)))
		return self.T_camera_aruco[cam_id]
	
	def correct_pose(self, data_dir, id, include_top = False, save_pose = False):
		T_camera_aruco = self.get_camera_aruco(id)
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
			if obj_id < 0 or obj_id >= len(self.obj_filename_list):
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
			start_id = (0 if include_top else 1)
			for obj_id in standard_model_list:
				T_camera_object = []
				for i in range(start_id, self.perspective_num):
					base_pose_dir = os.path.join(data_dir, str(i), 'pose')
					if '{}.npy'.format(obj_id) not in os.listdir(base_pose_dir):
						continue
					T_camera_base_aruco = self.get_camera_aruco(i)
					T_camera_base_object = np.load(os.path.join(base_pose_dir, '{}.npy'.format(obj_id)))
					T_camera_object.append(T_camera_aruco.dot(np.linalg.inv(T_camera_base_aruco)).dot(T_camera_base_object))
				if T_camera_object != []:
					T_camera_object = np.average(np.stack(T_camera_object), axis = 0, weights = self.perspective_pair_weight[id, start_id:])
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
	
	def correct_scene_pose(self, data_dir, include_top = False):
		for id in range(self.perspective_num):
			self.correct_pose(data_dir, id, include_top = include_top, save_pose = True)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', default = 'data', help = 'data for visualization', type = str)
	parser.add_argument('--id', default = 0, help = 'the perspective ID', type = int)
	parser.add_argument('--object_file_name_list', default = 'object_file_name_list.txt', help = 'ascii text file name that specifies the filenames of all possible objects', type = str)
	parser.add_argument('--weight_path', default = None, help = 'the path to the corrected weight, by default the matrix is set to a single-valued matrix.')
	FLAGS = parser.parse_args()
	corrector = PoseCorrector(object_file_name_list = FLAGS.object_file_name_list, perspective_num = 240, perspective_pair_weight_path = FLAGS.weight_path)
	_, _ = corrector.correct_pose(FLAGS.data_dir, int(FLAGS.id), save_pose = True)
