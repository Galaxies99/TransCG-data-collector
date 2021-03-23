import os
import argparse
import netrelay.client_pstrest as client

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
cmd_open = 'curl -X GET http://localhost:7278/PSTapi/StartTrackerDataStream'
cmd_close = 'curl --request POST --data \'\' http://localhost:7278/PSTapi/CloseDataStream'

while True:
    res, err = client.exec_cmd(s, cmd_open)
    print(res)
    str = input('Finish getting tracker data? (y/n): ')
    if str == 'y':
        break

with open('results/{}.json'.format(filename), 'w') as fres:
    fres.write(res)

os.system('python json_formatting.py --id {} --time {}'.format(ID, TIME))

# res, err = client.exec_cmd(s, cmd_close)
# print(res)

client.close(s)

os.system('python annotator.py --id {} --time {}'.format(ID, TIME))

os.system('python calc_transform.py --id {} --time {}'.format(ID, TIME))