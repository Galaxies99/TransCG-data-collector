import os
import sys
import argparse
from cv2 import cv2
import numpy as np
from .marker_calibrate import aruco_detector
sys.path.append(os.path.dirname(sys.path[0]))
from camera.camera import RealSenseCamera # pylint: disable=import-error


parser = argparse.ArgumentParser()
parser.add_argument('--id', default = 0, help = 'calibration id', type = int)
parser.add_argument('--path', default = 'robot_images', help = 'path of calibration images', type = str)
FLAGS = parser.parse_args()
FILE_PATH = FLAGS.path
ID = FLAGS.id

camera = RealSenseCamera(type='D435')

if os.path.exists(FILE_PATH) == False:
    os.mkdir(FILE_PATH)

print('********* Calibration Times {} Start *********'.format(ID))

img, _ = camera.get_full_image()
# Camera
MARKER_LENGTH = 150
CAMERA_INSTRINCS = np.load('../configs/camInstrincs.npy')
DIST_COEFFICIENTS = np.array([0., 0., 0., 0.]).reshape(4, 1)
IMG_PATH = os.path.join(FILE_PATH, 'img.png')

mat = aruco_detector(MARKER_LENGTH, CAMERA_INSTRINCS, DIST_COEFFICIENTS, IMG_PATH, print_flag=False, vis=False)
T_camera_pose = np.array(mat[0])

IMG_STORE_PATH = os.path.join(FILE_PATH, str(ID))
if os.path.exists(IMG_STORE_PATH) == False:
    os.mkdir(IMG_STORE_PATH)
cv2.imwrite(os.path.join(IMG_STORE_PATH, 'img.png'), img)

if os.path.exists('../configs/robot_calibration/') == False:
    os.makedirs('../configs/robot_calibration/')

with open(os.path.join('../configs/robot_calibration/', '{}.npy'.format(ID)), 'wb') as fT:
    np.save(fT, T_camera_pose)


print('********* Calibration Times {} Finished *********'.format(ID))
print('The pose matrix is ', T_camera_pose)

