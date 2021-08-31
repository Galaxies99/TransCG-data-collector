import json
import argparse
import numpy as np
from netrelay.utils_pstrest import postprocessing_data


def formatter_str(s):
    s = postprocessing_data(s)
    dic = json.loads(s)
    return dic


def formatter(filename):
    with open(filename, 'r') as ori_file:
        data = ori_file.read()
    dic = formatter_str(data)
    with open(filename, 'w') as res_file:
        json.dump(dic, res_file)


def find_obj(js, obj_name):
    js = js['TrackerData']['TargetPoses']
    obj_found = False
    for sub_js in js:
        if sub_js['TargetPose']['name'] == obj_name:
            T = np.array(sub_js['TargetPose']['TransformationMatrix']).reshape(4, 4)
            obj_found = True
    if not obj_found:
        raise ValueError('Object not found!')  
    return T


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', default='', help='json filename', type=str)
    FLAGS = parser.parse_args()
    filename = FLAGS.filename
    formatter(filename)