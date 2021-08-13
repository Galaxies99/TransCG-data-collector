import os
import cv2
import numpy as np
import copy
import trimesh
import argparse
import pyrender


os.environ['PYOPENGL_PLATFORM'] = 'egl'


class SceneRenderer(object):
    '''
    Render the depth image of the scene.
    '''
    def __init__(self, **kwargs):
        super(SceneRenderer, self).__init__()
        object_file_name_list = kwargs.get('object_file_name_list', 'object_file_name_list.txt')
        self.model_dir = kwargs.get('model_dir', 'models')
        self.perspective_num = kwargs.get('perspective_num', 240)
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
            self.models[model_id] = pyrender.Mesh.from_trimesh(trimesh.load(os.path.join(self.model_dir, obj_filename)))
        return self.models[model_id]

    def render_image(self, image_path, use_corrected_pose = False, save_result = True, epsilon = 1e-6, scale_factor = 1000):
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
            scene.add_node(node)
        
        height, width = original_depth.shape
        renderer = pyrender.OffscreenRenderer(viewport_width = width, viewport_height = height, point_size = 1.0)
        full_depth = renderer.render(scene, flags = pyrender.constants.RenderFlags.DEPTH_ONLY)
        full_depth = np.where(full_depth <= epsilon, 65535, full_depth * scale_factor)
        original_depth = np.where(original_depth <= epsilon * scale_factor, 65535, original_depth)
        depth = np.minimum(full_depth, original_depth).astype(np.uint16)
        depth = np.where(depth >= 65535 - epsilon * scale_factor, 0, depth)
        if save_result:
            cv2.imwrite(os.path.join(image_path, "depth1-gt.png"), depth)
        return depth
    
    def render_scene(self, scene_path, use_corrected_pose = False, epsilon = 1e-6, scale_factor = 1000):
        '''
        Render a scene, which contains several images.
        '''
        for image_id in range(self.perspective_num):
            self.render_image(os.path.join(scene_path, str(image_id)), use_corrected_pose = use_corrected_pose, save_result = True, epsilon = epsilon, scale_factor = scale_factor)


class SceneRenderer_L515(object):
    '''
    Render the depth image of the scene for L515 camera.
    '''
    def __init__(self, **kwargs):
        super(SceneRenderer_L515, self).__init__()
        object_file_name_list = kwargs.get('object_file_name_list', 'object_file_name_list.txt')
        self.model_dir = kwargs.get('model_dir', 'models')
        self.perspective_num = kwargs.get('perspective_num', 240)
        self.T_camera1_camera2 = np.load(kwargs.get('camera_transformation_path', os.path.join('configs', 'T_camera1_camera2.npy')))
        self.T_camera2_camera1 = np.linalg.inv(self.T_camera1_camera2)
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
            self.models[model_id] = pyrender.Mesh.from_trimesh(trimesh.load(os.path.join(self.model_dir, obj_filename)))
        return self.models[model_id]

    def render_image(self, image_path, use_corrected_pose = False, save_result = True, epsilon = 1e-6, scale_factor = 1000):
        '''
        Render a single image.
        '''
        scene = pyrender.Scene(ambient_light = [0.02, 0.02, 0.02], bg_color = [1.0, 1.0, 1.0])
        original_depth = np.array(cv2.imread(os.path.join(image_path, 'depth2.png'), cv2.IMREAD_UNCHANGED))
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
            obj_pose = self.T_camera2_camera1.dot(obj_pose)
            node = pyrender.Node(mesh = copy.deepcopy(self.get_models(obj_id)), matrix = obj_pose)
            nodes.append(node)
            scene.add_node(node)
        
        height, width = original_depth.shape
        renderer = pyrender.OffscreenRenderer(viewport_width = width, viewport_height = height, point_size = 1.0)
        full_depth = renderer.render(scene, flags = pyrender.constants.RenderFlags.DEPTH_ONLY)
        full_depth = np.where(full_depth <= epsilon, 65535, full_depth * scale_factor)
        original_depth = np.where(original_depth <= epsilon * scale_factor, 65535, original_depth)
        depth = np.minimum(full_depth, original_depth).astype(np.uint16)
        depth = np.where(depth >= 65535 - epsilon * scale_factor, 0, depth)
        if save_result:
            cv2.imwrite(os.path.join(image_path, "depth2-gt.png"), depth)
        return depth
    
    def render_scene(self, scene_path, use_corrected_pose = False, epsilon = 1e-6, scale_factor = 1000):
        '''
        Render a scene, which contains several images.
        '''
        for image_id in range(self.perspective_num):
            self.render_image(os.path.join(scene_path, str(image_id)), use_corrected_pose = use_corrected_pose, save_result = True, epsilon = epsilon, scale_factor = scale_factor)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', default = 'data/scene1/0/', help = 'data for rendering (should be a perspective of a scene)', type = str)
    parser.add_argument('--model_dir', default = 'models', help = 'ply model files directory path', type = str)
    parser.add_argument('--object_file_name_list', default = 'object_file_name_list.txt', help = 'ascii text file name that specifies the filenames of all possible objects', type = str)
    parser.add_argument('--corrected', action = 'store_true', help = 'whether to use the corrected poses.')
    FLAGS = parser.parse_args()
    renderer = SceneRenderer(object_file_name_list = FLAGS.object_file_name_list, model_dir = FLAGS.model_dir)
    renderer.render_image(image_path = FLAGS.image_path, use_corrected_pose = FLAGS.corrected, save_result = True)