import os
import argparse
from tqdm import tqdm
from .depth_renderer import SceneRenderer_D435, SceneRenderer_L515
from .sn_generator import SurfaceNormalGenerator

os.environ['PYOPENGL_PLATFORM'] = 'egl'
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default = 'data', help = 'data for postprocessing', type = str)
parser.add_argument('--perspective_num', default = 240, help = 'the perspective number', type = int)
parser.add_argument('--model_dir', default = 'models', help = 'ply model files directory path', type = str)
parser.add_argument('--begin_id', default = 1, help = 'begin scene id', type = int)
parser.add_argument('--end_id', default = 130, help = 'end scene id', type = int)
parser.add_argument('--object_file_name_list', default = 'object_file_name_list.txt', help = 'ascii text file name that specifies the filenames of all possible objects', type = str)
parser.add_argument('--camera_transformation_path', default = os.path.join('configs', 'T_camera2_camera1.npy'), help = 'the path to the transformation matrix between cameras', type = str)
FLAGS = parser.parse_args()
begin_id = int(FLAGS.begin_id)
end_id = int(FLAGS.end_id)
renderer_D435 = SceneRenderer_D435(object_file_name_list = FLAGS.object_file_name_list, model_dir = FLAGS.model_dir, perspective_num = FLAGS.perspective_num)
renderer_L515 = SceneRenderer_L515(object_file_name_list = FLAGS.object_file_name_list, model_dir = FLAGS.model_dir, perspective_num = FLAGS.perspective_num, camera_transformation_path = FLAGS.camera_transformation_path)
for id in tqdm(range(begin_id, end_id + 1)):
    renderer_D435.render_scene(os.path.join(FLAGS.data_dir, 'scene{}'.format(id)), use_corrected_pose = True)
    renderer_L515.render_scene(os.path.join(FLAGS.data_dir, 'scene{}'.format(id)), use_corrected_pose = True)

sn_gen = SurfaceNormalGenerator('data', display_log = False)
for id in tqdm(range(begin_id, end_id + 1)):
    sn_gen.gen_scene_normals(id)