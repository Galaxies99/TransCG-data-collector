import os
import cv2
import json
import argparse
import numpy as np
from PIL import Image


class DataWasher(object):
    """
    Surface normal generator for the dataset: given an depth image, generate its surface normal.
    """
    def __init__(self, data_path, **kwargs):
        super(DataWasher, self).__init__()
        self.data_path = data_path
        self.epsilon = kwargs.get('epsilon', 0.2)
    
    def wash_scene(self, scene_id):
        scene_path = os.path.join(self.data_path, "scene{}".format(scene_id))
        with open(os.path.join(scene_path, 'metadata.json'), 'r') as fp:
            scene_metadata = json.load(fp)
        rm_num = 0
        modified1 = False
        rm_list = []
        for image_id in scene_metadata['D435_valid_perspective_list']:
            depth = np.array(Image.open(os.path.join(scene_path, str(image_id), 'depth1.png')))
            if depth.max() < self.epsilon:
                modified1 = True
                rm_list.append(image_id)
        if modified1:
            for id in rm_list:
                scene_metadata['D435_valid_perspective_list'].remove(id)
            scene_metadata['D435_valid_perspective_num'] = len(scene_metadata['D435_valid_perspective_list'])
            rm_num += len(rm_list)
            rm_list = []
        
        modified2 = False
        for image_id in scene_metadata['L515_valid_perspective_list']:
            depth = np.array(Image.open(os.path.join(scene_path, str(image_id), 'depth2.png')))
            if depth.max() < self.epsilon:
                modified2 = True
                rm_list.append(image_id)
        if modified2:
            for id in rm_list:
                scene_metadata['L515_valid_perspective_list'].remove(id)
            scene_metadata['L515_valid_perspective_num'] = len(scene_metadata['L515_valid_perspective_list'])
            rm_num += len(rm_list)
        if modified1 or modified2:
            print('Scene {} has invalid images, remove {} images.'.format(scene_id, rm_num))
            with open(os.path.join(scene_path, 'metadata.json'), 'w') as fp:
                json.dump(scene_metadata, fp)
        else:
            print('Scene {} clean.'.format(scene_id))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default = 'data', help = 'data for visualization', type = str)
    parser.add_argument('--begin_id', default = 1, help = 'begin scene id', type = int)
    parser.add_argument('--end_id', default = 130, help = 'end scene id', type = int)
    FLAGS = parser.parse_args()
    washer = DataWasher(FLAGS.data_dir, display_log = True)
    begin_id = int(FLAGS.begin_id)
    end_id = int(FLAGS.end_id)
    for id in range(begin_id, end_id + 1):
        washer.wash_scene(id)
