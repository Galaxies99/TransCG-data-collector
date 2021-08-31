import os
import argparse
import numpy as np
import netrelay.client as client
from utils.pst_jsonhandler import formatter_str, find_obj

parser = argparse.ArgumentParser()
parser.add_argument('--id', default = 0, help = 'Object ID', type = int)
parser.add_argument('--time', default = 0, help = 'Times of sampling', type = int)
parser.add_argument('--ip', default = '10.52.25.177', help = 'IP address of the computer with Windows system', type = str)
parser.add_argument('--port', default = 23333, help = 'The port that are used in netrelay', type = int)
parser.add_argument('--object_file_name_list', default = 'object_file_name_list.txt', help = 'ascii text file name that specifies the filenames of all possible objects', type = str)
FLAGS = parser.parse_args()
ID = FLAGS.id
TIME = FLAGS.time
IP = FLAGS.ip
PORT = FLAGS.port
OBJECT_FILE_NAME_LIST_FILE_NAME = FLAGS.object_file_name_list

objectfilenamelist = []
with open(OBJECT_FILE_NAME_LIST_FILE_NAME, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if not (line == '\n'):
            objectfilenamelist.append(line.replace('\n', '').replace('\r', ''))


filename = '{}-{}'.format(ID, TIME)
obj_name, _ = os.path.splitext(objectfilenamelist[ID])
print(obj_name)

s, id = client.start((IP, PORT))
cmd_tracker = 'GetTracker'

T_list = []

while True:
    res = client.exec_cmd(s, cmd_tracker)
    print(res)
    str = input('Collect this tracker data? (y - yes, n - no, f - finish): ')
    if str == 'f':
        break
    elif str == 'n':
        continue
    elif str != 'y':
        input('Invalid input, press enter to re-collect data.')
        continue
    js = formatter_str(res)
    try:
        T = find_obj(js, obj_name)
    except ValueError:
        input('Object not found, press enter to re-collect data.')
        continue
    T_list.append(T)

T_list = np.array(T_list)
T = np.mean(T_list, axis=0)

with open('results/{}.npy'.format(filename), 'wb') as fres:
    np.save(fres, T)

client.close(s)

os.system('python annotation/annotator.py --id {} --time {}'.format(ID, TIME))

os.system('python annotation/calc_transform.py --id {} --time {}'.format(ID, TIME))