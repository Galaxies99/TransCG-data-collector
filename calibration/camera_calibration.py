import os
import sys
import argparse
from cv2 import cv2
import numpy as np
from marker_calibrate import aruco_detector
sys.path.append(os.path.dirname(sys.path[0]))
from camera.camera import RealSenseCamera # pylint: disable=import-error

parser = argparse.ArgumentParser()
parser.add_argument('--times', default=5, help='Calibration times', type=int)
parser.add_argument('--path', default=os.path.join('data', 'calibration', 'camera_calib_images'), help='path of calibration images', type=str)
FLAGS = parser.parse_args()
TIMES = FLAGS.times
FILE_PATH = FLAGS.path

camera1 = RealSenseCamera(type='D435')
camera2 = RealSenseCamera(type='L515')

if os.path.exists(FILE_PATH) == False:
    os.makedirs(FILE_PATH)

T_camera2_camera1_list = []

for i in range(TIMES):
    print('********* Calibration Times {} Start *********'.format(i))
    
    image1, _ = camera1.get_full_image()
    image2, _ = camera2.get_full_image()
    cv2.imwrite(os.path.join(FILE_PATH, 'img1.png'), image1)
    cv2.imwrite(os.path.join(FILE_PATH, 'img2.png'), image2)
	
    MARKER_LENGTH = 150
    CAMERA_INSTRINCS = np.load('../configs/camInstrincs.npy')
    DIST_COEFFICIENTS = np.array([0., 0., 0., 0.]).reshape(4, 1)
    IMG_PATH = os.path.join(FILE_PATH, 'img1.png')

    mat = aruco_detector(MARKER_LENGTH, CAMERA_INSTRINCS, DIST_COEFFICIENTS, IMG_PATH, print_flag=False, vis=False)
    T_camera1_calibration = mat[0]

    MARKER_LENGTH = 150
    CAMERA_INSTRINCS = np.load('../configs/camInstrincs-L515.npy')
    DIST_COEFFICIENTS = np.array([0., 0., 0., 0.]).reshape(4, 1)
    IMG_PATH = os.path.join(FILE_PATH, 'img2.png')
    
    mat = aruco_detector(MARKER_LENGTH, CAMERA_INSTRINCS, DIST_COEFFICIENTS, IMG_PATH, print_flag=False, vis=False)
    T_camera2_calibration = mat[0]

    T_camera2_camera1 = T_camera2_calibration.dot(np.linalg.inv(T_camera1_calibration))
    print('********* Calibration Times {} Finished *********'.format(i))
    print('The transformation matrix from camera2 to camera1 is: ', T_camera2_camera1)
    T_camera2_camera1_list.append(T_camera2_camera1)
    
    IMG_STORE_PATH = os.path.join(FILE_PATH, str(i))
    if os.path.exists(IMG_STORE_PATH) == False:
        os.mkdir(IMG_STORE_PATH)
    cv2.imwrite(os.path.join(IMG_STORE_PATH, 'img1.png'), image1)
    cv2.imwrite(os.path.join(IMG_STORE_PATH, 'img2.png'), image2)

    if i != TIMES - 1:
        input('Press Enter to continue ...')


T_camera2_camera1_list = np.array(T_camera2_camera1_list)
T_camera2_camera1 = np.mean(T_camera2_camera1_list, axis=0).reshape(4, 4)

print('********* Calibration Results *********')
print('The transformation matrix from camera2 to camera1 is: ', T_camera2_camera1)
with open('../configs/T_camera2_camera1.npy', 'wb') as fT:
    np.save(fT, T_camera2_camera1)


