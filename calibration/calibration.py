import os
import sys
import cv2
import argparse
import numpy as np
from marker_calibrate import aruco_detector
sys.path.append(os.path.dirname(sys.path[0]))
from camera.camera import RealSenseCamera
import netrelay.client_pstrest as client
from jsonhandler import formatter_str, find_obj


parser = argparse.ArgumentParser()
parser.add_argument('--ip', default='10.52.25.177', help='IP address of the computer with Windows system', type=str)
parser.add_argument('--port', default=23333, help='The port that are used in netrelay', type=int)
FLAGS = parser.parse_args()
IP = FLAGS.ip
PORT = FLAGS.port

camera = RealSenseCamera()
def img_from_cam():
	image, image_depth = camera.get_rgbd()
	image = (image * 255).astype(np.uint8)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	return image, image_depth
    
image, _ = img_from_cam()
cv2.imwrite('img.png', image)

MARKER_LENGTH = 150
CAMERA_INSTRINCS = np.load('../configs/camInstrincs.npy')
DIST_COEFFICIENTS = np.array([0., 0., 0., 0.]).reshape(4, 1)
IMG_PATH = 'img.png'

mat = aruco_detector(MARKER_LENGTH, CAMERA_INSTRINCS, DIST_COEFFICIENTS, IMG_PATH, print_flag=False, vis=False)
T_camera_calibration = mat[0]

s, id = client.start((IP, PORT))
cmd_open = 'curl -X GET http://localhost:7278/PSTapi/StartTrackerDataStream'
cmd_close = 'curl --request POST --data \'\' http://localhost:7278/PSTapi/CloseDataStream'

while True:
    res, err = client.exec_cmd(s, cmd_open)
    print(res)
    str = input('Finish getting tracker data? (y/n): ')
    if str == 'y':
        break

js = formatter_str(res)

# res, err = client.exec_cmd(s, cmd_close)
# print(res)

client.close(s)

T_tracker_calibration = find_obj(js, 'calibration')

T = T_tracker_calibration.dot(np.linalg.inv(T_camera_calibration))
print('The transformation matrix from tracker to camera is: ', T)

with open('../configs/T_tracker_camera.npy', 'wb') as fT:
    np.save(fT, T)
