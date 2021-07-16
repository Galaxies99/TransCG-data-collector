import os
import cv2
import numpy as np
import copy
import trimesh
import pyrender
from model import loadmodel


class SceneRenderer(object):
    '''
    Render the depth image of the scene.
    '''
    def __init__(self, **kwargs):
        super(SceneRenderer, self).__init__()
        object_file_name_list = kwargs.get('object_file_name_list', 'object_file_name_list.txt')
        self.model_dir = kwargs.get('model_dir', 'models')
        with open(object_file_name_list, 'r') as object_filename_file:
            self.obj_filename_list = []
            for line in object_filename_file.readlines():
                if not (line == '\n'):
                    self.obj_filename_list.append(line.replace('\n', '').replace('\r', ''))
        
        self.models = []
        for _ in self.obj_filename_list:
            self.models.append(None)
    
    def get_models(self, model_id):
        '''
        Get the corresponding models.
        '''
        if model_id < 0 or model_id >= len(self.obj_filename_list):
            raise ValueError('model id out of range.')
        obj_filename = self.obj_filename_list[model_id]
        if self.models[model_id] is None:
            self.models[model_id] = loadmodel(self.model_dir, obj_filename)
        return self.models[model_id]

    def render_image(self, image_path, use_corrected_pose = False):
        '''
        Render a single image.
        '''
        scene = pyrender.Scene(ambient_light = [0.02, 0.02, 0.02], bg_color = [1.0, 1.0, 1.0])
        original_depth = np.array(cv2.imread(os.path.join(image_path, 'depth1.png'), cv2.IMREAD_UNCHANGED))
        cam = pyrender.IntrinsicsCamera(927.17, 927.37, 651.32, 349.62)
        flip = np.eye(4)
        flip[1, 1] = flip[2, 2] = -1
        scene.add(cam, pose = flip)

        pose_dir = os.path.join(image_path, 'corrected_pose' if use_corrected_pose else 'pose')
        obj_list = []
        for filename in os.listdir(pose_dir):
            obj_id, ext = os.path.splitext(filename)
            if ext != '.npy':
                continue
            try:
                obj_id = int(obj_id)
            except Exception:
                continue
            if obj_id < 0 or obj_id >= len(self.obj_filename_list):
                continue
            obj_list.append(obj_id)
        
        nodes = []
        for obj_id in obj_list:
            obj_pose = np.load(os.path.join(pose_dir, '{}.npy'.format(obj_id))) 
            node = pyrender.Node(mesh = copy.deepcopy(self.get_models(obj_id)), matrix = obj_pose)
            nodes.append(node)
            scene.add(nodes)
        
        width, height = original_depth.shape
        renderer = pyrender.OffscreenRenderer(viewport_width = width, viewport_height = height, point_size = 1.0)
        full_depth = renderer.render(scene, flags = pyrender.constants.RenderFlags.DEPTH_ONLY)

        return full_depth

        


if __name__ == '__main__':
    renderer = SceneRenderer()
    depth = renderer.render_image('data/scene1/0/')
    depth = depth / depth.max()
    cv2.imshow('depth', depth)
    cv2.waitKey(0)