import os
import json
import argparse
import netrelay.client as client
from jsonhandler import formatter_str

parser = argparse.ArgumentParser()
parser.add_argument('--id', default=0, help='Object ID', type=int)
parser.add_argument('--time', default=0, help='Times of sampling', type=int)
parser.add_argument('--ip', default='10.52.25.177', help='IP address of the computer with Windows system', type=str)
parser.add_argument('--port', default=23333, help='The port that are used in netrelay', type=int)
FLAGS = parser.parse_args()
ID = FLAGS.id
TIME = FLAGS.time
IP = FLAGS.ip
PORT = FLAGS.port

filename = '{}-{}'.format(ID, TIME)

s, id = client.start((IP, PORT))
cmd_tracker = 'GetTracker'

while True:
    res = client.exec_cmd(s, cmd_tracker)
    print(res)
    str = input('Finish getting tracker data? (y/n): ')
    if str == 'y':
        break

with open('results/{}.json'.format(filename), 'w') as fres:
    res = formatter_str(res)
    json.dump(res, fres)

client.close(s)

os.system('python annotator.py --id {} --time {}'.format(ID, TIME))

os.system('python calc_transform.py --id {} --time {}'.format(ID, TIME))