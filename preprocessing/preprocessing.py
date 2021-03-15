import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--server', default='~/hongjie/meshlab/distrib/meshlabserver', help='The path to the meshlabserver (usually in [Your MeshLab path]/distrib/meshlabserver)')
FLAGS = parser.parse_args()
meshlabserver_dir = FLAGS.server

obj_name = []
with open('../object_file_name_list.txt', 'r') as file:
    for line in file.readlines():
        if not (line == '\n'): 
            name, _ = os.path.splitext(line.replace('\n', '').replace('\r', ''))
            obj_name.append(name)

for name in obj_name:
    os.system(meshlabserver_dir + ' -i rawdata/{}/{}.obj -o rawdata/temp/{}.ply -m vc vf vn fc ff fn -s trans_t2v.mlx'.format(name, name, name))
    os.system(meshlabserver_dir + ' -i rawdata/temp/{}.ply -o ../models/{}.ply -m vc vf vn fc ff fn -s sim_mesh.mlx'.format(name, name, name))

