import os
import argparse
from tqdm import tqdm
from pose_corrector import PoseCorrector
from depth_renderer import SceneRenderer


os.environ['PYOPENGL_PLATFORM'] = 'egl'
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default = 'data', help = 'data for postprocessing', type = str)
parser.add_argument('--model_dir', default = 'models', help = 'ply model files directory path', type = str)
parser.add_argument('--begin_id', default = 1, help = 'begin scene id', type = int)
parser.add_argument('--end_id', default = 100, help = 'end scene id', type = int)
parser.add_argument('--object_file_name_list', default = 'object_file_name_list.txt', help = 'ascii text file name that specifies the filenames of all possible objects', type = str)
parser.add_argument('--weight_path', default = None, help = 'the path to the corrected weight, by default the matrix is set to a single-valued matrix.')
FLAGS = parser.parse_args()
begin_id = int(FLAGS.begin_id)
end_id = int(FLAGS.end_id)
corrector = PoseCorrector(object_file_name_list = FLAGS.object_file_name_list, perspective_num = 240, perspective_pair_weight_path = FLAGS.weight_path)
renderer = SceneRenderer(object_file_name_list = FLAGS.object_file_name_list, model_dir = FLAGS.model_dir)
for id in tqdm(range(begin_id, end_id + 1)):
    corrector.correct_scene_pose(os.path.join(FLAGS.data_dir, 'scene{}'.format(id)))
    renderer.render_scene(os.path.join(FLAGS.data_dir, 'scene{}'.format(id)), use_corrected_pose = True)
