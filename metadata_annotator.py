"""
Meta-data annotator.

Author: Hongjie Fang
"""

import os
import cv2
import json
import argparse
import numpy as np
from pynput import keyboard
from model import loadmodel
from renderer import draw_model

from pose_corrector import PoseCorrector


class MetadataAnnotator(object):
    """
    Meta-data annotator for scenes.
    """
    def __init__(
        self, 
        data_path = 'data', 
        scene_id = 0, 
        perspective_num = 240,
        model_path = 'models', 
        object_file_name_list = 'object_file_name_list.txt',
        camera_calib_file = os.path.join('data', 'calibration', 'T_camera2_camera1.npy'),
        corrected = True,
        weight_path = None,
        **kwargs
    ):
        """
        Initialization.

        Parameters
        ----------
        data_path: str, optional, default: 'data', the path to the data directory;
        scene_id: int, optional, default: 0, the scene ID;
        perspective_num: int, optional, default: 240, the perspective number;
        model_path: str, optional, default: 'models', the path to the model directory;
        object_file_name_list: str, optional, default: 'object_file_name_list.txt', the object file name list;
        camera_calib_file: str, optional, default: 'data/calibration/T_camera2_camera1.npy', camera calibration file in npy format;
        corrected: bool, optional, default: True, whether or not correct the pose;
        weight_path: str, optional, default: None, the path to the weight information for pose correction (setting None to use the matrix with all 1s; only need to be set when corrected is True).
        """
        super(MetadataAnnotator, self).__init__()
        self.scene_path = os.path.join(data_path, 'scene{}'.format(scene_id))
        self.model_path = model_path
        with open(object_file_name_list, 'r') as object_filename_file:
            self.obj_filename_list = []
            for line in object_filename_file.readlines():
                if not (line == '\n'):
                    self.obj_filename_list.append(line.replace('\n', '').replace('\r', ''))
        self.num_models = len(self.obj_filename_list)
        self.models = [None] * self.num_models
        self.T_camera2_camera1 = np.load(camera_calib_file)
        self.perspective_num = perspective_num
        self.corrected = corrected
        if self.corrected:
            print("[Log] Automatically correcting poses ...")
            self.pose_corrector = PoseCorrector(object_file_name_list = object_file_name_list, perspective_num = perspective_num, perspective_pair_weight_path = weight_path)
            self.pose_corrector.correct_scene_pose(self.scene_path, include_top = False)
            self.pose_dir = "corrected_pose"
            print("[Log] Pose correction finished!")
        else:
            self.pose_dir = "pose"
        self.scene_model_list = self.get_scene_models()
        self.metadata_file = os.path.join(self.scene_path, "metadata.json")
        self.has_metadata = os.path.exists(self.metadata_file)
        self.validation = [[False] * self.perspective_num, [False] * self.perspective_num]
        if self.has_metadata:
            print('[Log] Metadata found, use annotated metadata.')
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
            D435_list = self.metadata['D435_valid_perspective_list']
            for id in D435_list:
                self.validation[0][id] = True
            L515_list = self.metadata['L515_valid_perspective_list']
            for id in L515_list:
                self.validation[1][id] = True
    
    def get_models(self, model_id):
        """
        Get the corresponding models.

        Parameters
        ----------
        model_id: the model ID.
        """
        if model_id < 0 or model_id >= self.num_models:
            raise ValueError('model id out of range.')
        obj_filename = self.obj_filename_list[model_id]
        if self.models[model_id] is None:
            self.models[model_id] = loadmodel(self.model_path, obj_filename)
        return self.models[model_id]
    
    def get_scene_models(self):
        """
        Get standard models of the scene.
        """
        standard_pose_dir = os.path.join(self.scene_path, '0', 'pose')
        scene_model_list = []
        for filename in os.listdir(standard_pose_dir):
            obj_id, ext = os.path.splitext(filename)
            if ext != '.npy':
                continue
            try:
                obj_id = int(obj_id)
            except Exception:
                continue
            if obj_id < 0 or obj_id >= self.num_models:
                continue
            scene_model_list.append(obj_id)
        return scene_model_list
    
    def get_rendered_image(self, perspective_id, image_id):
        """
        Get the rendered image.

        Parameters
        ----------
        perspective_id: the perspective ID;
        image_id: the image ID, 1 for D435 image, 2 for L515 image.
        """
        image = cv2.imread(os.path.join(self.scene_path, str(perspective_id), 'rgb{}.png'.format(image_id)))
        rendered_image = image
        if image_id == 1:
            cam = np.load(os.path.join('configs', 'camIntrinsics-D435.npy'))
        else:
            cam = np.load(os.path.join('configs', 'camIntrinsics-L515.npy'))
        for obj_id in self.scene_model_list:
            pose_file = os.path.join(self.scene_path, str(perspective_id), self.pose_dir, '{}.npy'.format(obj_id))
            if os.path.exists(pose_file):
                model = self.get_models(obj_id)
                T_camera_object = np.load(pose_file)
                if image_id == 2:
                    T_camera_object = np.matmul(self.T_camera2_camera1, T_camera_object)
                rendered_image = draw_model(rendered_image, T_camera_object, cam, model)
        return image, rendered_image
    
    def _on_press(self, key):
        try:
            if key == keyboard.Key.enter:
                if self.finish:
                    print('[Log] Annotation saved.')
                    self.loop = False
                else:
                    print('[Log] Annotation has not finished yet, please continue performing annotations.')                
            elif key.char == 'y':
                if not self.switching and self.cur_perspective_id < self.perspective_num:
                    self.validation[self.cur_camera_id][self.cur_perspective_id] = True
                    if self.cur_perspective_id != self.perspective_num - 1:
                        self.cur_perspective_id += 1
                        self.switching = True
                    else:
                        self.finish = True
                    self.max_perspective_id = max(self.cur_perspective_id, self.max_perspective_id)
            elif key.char == 'n':
                if not self.switching and self.cur_perspective_id < self.perspective_num:
                    self.validation[self.cur_camera_id][self.cur_perspective_id] = False
                    if self.cur_perspective_id != self.perspective_num - 1:
                        self.cur_perspective_id += 1
                        self.switching = True
                    else:
                        self.finish = True
                    self.max_perspective_id = max(self.cur_perspective_id, self.max_perspective_id)
            elif key.char == 'a':
                if not self.switching and self.cur_perspective_id > 0:
                    self.switching = True
                    self.cur_perspective_id -= 1
            elif key.char == 'd':
                if not self.switching and self.cur_perspective_id < self.max_perspective_id:
                    self.switching = True
                    self.cur_perspective_id += 1
            elif key.char == '.':
                if self.transparency <= 0.9:
                    self.transparency += 0.1
            elif key.char == ',':
                if self.transparency >= 0.1:
                    self.transparency -= 0.1
            elif key.char == 'q':
                self.quit = True
                return False
        except Exception:
            pass

    def _on_release(self, key):
        pass

    def annotate(self, camera_id):
        """
        Annotation.

        Parameters
        ----------
        camera_id: the camera ID, 1 for D435, 2 for L515.
        """
        self.cur_perspective_id = 0
        if self.has_metadata or camera_id == 2:
            self.max_perspective_id = self.perspective_num - 1
            self.finish = True
        else:
            self.max_perspective_id = 0
            self.finish = False
        self.cur_camera_id = camera_id - 1
        self.switching = False
        self.quit = False
        self.transparency = 0.5
        self.listener = keyboard.Listener(on_press = self._on_press, on_release = self._on_release)
        image, rendered_image = self.get_rendered_image(self.cur_perspective_id, camera_id)
        image_perspective_id = self.cur_perspective_id
        self.listener.start()
        font_size = 0.5
        font_thickness = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_color = (255, 0, 255)
        self.loop = True
        while self.loop and not self.quit:
            if self.cur_perspective_id != image_perspective_id:
                print('[Log] Current annotating image {}, current progress: {}/{}'.format(self.cur_perspective_id + 1 + self.perspective_num * (camera_id - 1), self.max_perspective_id + 1 + self.perspective_num * (camera_id - 1), self.perspective_num * 2))
                image, rendered_image = self.get_rendered_image(self.cur_perspective_id, camera_id)
                image_perspective_id = self.cur_perspective_id
                self.switching = False
            final = (rendered_image * self.transparency + image * (1 - self.transparency)).astype(np.uint8)
            final = cv2.putText(final, 'Transparency: %.1f' % self.transparency, (20, final.shape[0] - 10), font, font_size, font_color, font_thickness)
            if self.max_perspective_id == self.cur_perspective_id and not self.finish:
                final = cv2.putText(final, 'Image: {}; Current status: Not annotated.'.format(self.cur_perspective_id + 1 + self.perspective_num * (camera_id - 1)), (20, 25), font, font_size, font_color, font_thickness)
            else:
                final = cv2.putText(final, 'Image: {}; Current status: {}.'.format(self.cur_perspective_id + 1 + self.perspective_num * (camera_id - 1), 'Valid' if self.validation[camera_id - 1][self.cur_perspective_id] else 'Invalid'), (20, 25), font, font_size, font_color, font_thickness)
            cv2.imshow('Final Annotator', final)
            cv2.waitKey(1)
        self.listener.stop()
    
    def generate_metadata(self):
        """
        Generate metadata.
        """
        self.metadata = {
            "model_list": self.scene_model_list,
            "D435_valid_perspective_num": len(self.available[0]),
            "D435_valid_perspective_list": self.available[0],
            "L515_valid_perspective_num": len(self.available[1]),
            "L515_valid_perspective_list": self.available[1]
        }
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f)
        self.has_metadata = True
    
    def run(self):
        """
        Run the metadata annotation process.
        """
        self.annotate(camera_id = 1)
        if not self.has_metadata:
            print("[Log] Automatically load D435 annotation results as initial results of L515.")
            self.validation[1] = self.validation[0]
        self.annotate(camera_id = 2)
        self.available = [[], []]
        for i in range(self.perspective_num):
            if self.validation[0][i]:
                self.available[0].append(i)
            if self.validation[1][i]:
                self.available[1].append(i)
        print('[Log] Valid D435 images: {}, valid L515 images: {}'.format(len(self.available[0]), len(self.available[1])))
        if not self.quit:
            self.generate_metadata()
        else:
            print('[Log] Detect "q" is pressed, the metadata will not be saved.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default = 'models', help = 'ply model files directory path', type = str)
    parser.add_argument('--data_dir', default = 'data', help = 'data for visualization', type = str)
    parser.add_argument('--id', default = 0, help = 'the scene ID', type = int)
    parser.add_argument('--perspective_num', default = 240, help = 'the perspective number', type = int)
    parser.add_argument('--object_file_name_list', default = 'object_file_name_list.txt', help = 'ascii text file name that specifies the filenames of all possible objects', type = str)
    parser.add_argument('--camera_calibration_file', default = os.path.join('configs', 'T_camera2_camera1.npy'), help = 'the path to the camera calibration file in npy format', type = str)
    parser.add_argument('--corrected', action = 'store_true', help = 'whether to use the corrected poses.')
    parser.add_argument('--weight_path', default = None, help = 'the path to the corrected weight, by default the matrix is set to a single-valued matrix.')
    FLAGS = parser.parse_args()
    annotator = MetadataAnnotator(
        data_path = FLAGS.data_dir,
        scene_id = FLAGS.id,
        perspective_num = FLAGS.perspective_num,
        model_path = FLAGS.model_dir,
        object_file_name_list = FLAGS.object_file_name_list,
        camera_calib_file = FLAGS.camera_calibration_file,
        corrected = FLAGS.corrected,
        weight_path = FLAGS.weight_path
    )

    annotator.run()
