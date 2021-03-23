import json
import argparse
from netrelay.utils_pstrest import postprocessing_data

parser = argparse.ArgumentParser()
parser.add_argument('--id', default=0, help='Object ID', type=int)
parser.add_argument('--time', default=0, help='Times of sampling', type=int)
FLAGS = parser.parse_args()
ID = FLAGS.id
TIME = FLAGS.time

filename = 'results/{}-{}.json'.format(ID, TIME)

with open(filename, 'r') as ori_file:
    data = ori_file.read()

res = postprocessing_data(data)
dic = json.loads(res)

with open(filename, 'w') as res_file:
    json.dump(dic, res_file)