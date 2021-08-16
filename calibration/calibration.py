import os
import sys
import argparse
from cv2 import cv2
import numpy as np
from marker_calibrate import aruco_detector
sys.path.append(os.path.dirname(sys.path[0]))
from camera.camera import RealSenseCamera # pylint: disable=import-error
import netrelay.client as client # pylint: disable=import-error
from jsonhandler import formatter_str, find_obj # pylint: disable=import-error


parser = argparse.ArgumentParser()
parser.add_argument('--ip', default='10.52.25.177', help='IP address of the computer with Windows system', type=str)
parser.add_argument('--port', default=23333, help='The port that are used in netrelay', type=int)
parser.add_argument('--times', default=5, help='Calibration times', type=int)
parser.add_argument('--path', default='images', help='path of calibration images', type=str)
FLAGS = parser.parse_args()
IP = FLAGS.ip
PORT = FLAGS.port
TIMES = FLAGS.times
FILE_PATH = FLAGS.path

camera1 = RealSenseCamera(type='D435')
camera2 = RealSenseCamera(type='L515')

if os.path.exists(FILE_PATH) == False:
    os.mkdir(FILE_PATH)

s, id = client.start((IP, PORT))

T_tracker_camera1_list = []
T_camera1_camera2_list = []

for i in range(TIMES):
    print('********* Calibration Times {} Start *********'.format(i))
    
    cmd = 'GetTracker'

    while True:
        image1, _ = camera1.get_full_image()
        image2, _ = camera2.get_full_image()
        cv2.imwrite(os.path.join(FILE_PATH, 'img1.png'), image1)
        cv2.imwrite(os.path.join(FILE_PATH, 'img2.png'), image2)
        res = client.exec_cmd(s, cmd)
        print(res)
        ops = input('Finish getting tracker data? (y/n): ')
        if ops == 'y':
            break

    # Camera1 <-> Tracker
    MARKER_LENGTH = 150
    CAMERA_INSTRINCS = np.load('../configs/camInstrincs.npy')
    DIST_COEFFICIENTS = np.array([0., 0., 0., 0.]).reshape(4, 1)
    IMG_PATH = os.path.join(FILE_PATH, 'img1.png')

    mat = aruco_detector(MARKER_LENGTH, CAMERA_INSTRINCS, DIST_COEFFICIENTS, IMG_PATH, print_flag=False, vis=False)
    T_camera1_calibration = mat[0]

    js = formatter_str(res)
    
    T_tracker_calibration = find_obj(js, 'calibration')

    T_tracker_camera1 = T_tracker_calibration.dot(np.linalg.inv(T_camera1_calibration))
    print('********* Calibration Times {} Finished *********'.format(i))
    print('The transformation matrix from tracker to camera1 is: ', T_tracker_camera1)
    T_tracker_camera1_list.append(T_tracker_camera1)

    # Camera1 <-> Camera2
    MARKER_LENGTH = 150
    CAMERA_INSTRINCS = np.load('../configs/camInstrincs.npy')
    DIST_COEFFICIENTS = np.array([0., 0., 0., 0.]).reshape(4, 1)
    IMG_PATH = os.path.join(FILE_PATH, 'img2.png')
    
    mat = aruco_detector(MARKER_LENGTH, CAMERA_INSTRINCS, DIST_COEFFICIENTS, IMG_PATH, print_flag=False, vis=False)
    T_camera2_calibration = mat[0]

    T_camera1_camera2 = T_camera1_calibration.dot(np.linalg.inv(T_camera2_calibration))
    print('********* Calibration Times {} Finished *********'.format(i))
    print('The transformation matrix from camera1 to camera2 is: ', T_camera1_camera2)
    T_camera1_camera2_list.append(T_camera1_camera2)
    
    IMG_STORE_PATH = os.path.join(FILE_PATH, str(i))
    if os.path.exists(IMG_STORE_PATH) == False:
        os.mkdir(IMG_STORE_PATH)
    cv2.imwrite(os.path.join(IMG_STORE_PATH, 'img1.png'), image1)
    cv2.imwrite(os.path.join(IMG_STORE_PATH, 'img2.png'), image2)

    if i != TIMES - 1:
        input('Press Enter to continue ...')

client.close(s)

T_tracker_camera1_list = np.array(T_tracker_camera1_list)
T_tracker_camera1 = np.mean(T_tracker_camera1_list, axis=0).reshape(4, 4)
T_camera1_camera2_list = np.array(T_camera1_camera2_list)
T_camera1_camera2 = np.mean(T_camera1_camera2_list, axis=0).reshape(4, 4)

print('********* Calibration Results *********')
print('The transformation matrix from tracker to camera is: ', T_tracker_camera1)
print('The transformation matrix from camera1 to camera2 is: ', T_camera1_camera2)
with open('../configs/T_tracker_camera.npy', 'wb') as fT:
    np.save(fT, T_tracker_camera1)
with open('../configs/T_camera1_camera2.npy', 'wb') as fT:
    np.save(fT, T_camera1_camera2)


