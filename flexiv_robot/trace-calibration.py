import numpy as np
import time
import os
import cv2
from .flexiv import FlexivRobot
from camera.camera import RealSenseCamera
from calibration.marker_calibrate import aruco_detector

robot_ip_address = '192.168.2.100'
pc_ip_address = '192.168.2.200'
safe_duration = 5
normal_duration = 3
fast_duration = 1


def pause():
    print('Press enter to continue ...')
    input()

camera = RealSenseCamera(type='D435')

FILE_PATH = os.path.join('data', 'calibration', 'robot_images')

if os.path.exists(FILE_PATH) is False:
    os.makedirs(FILE_PATH) 

def calib(ID, DEBUG = False):
    img, _ = camera.get_full_image()
    cv2.imwrite(os.path.join(FILE_PATH, 'img.png'), img)
    # Camera
    MARKER_LENGTH = 150
    CAMERA_INSTRINCS = np.load(os.path.join('configs', 'camIntrinsics-D435.npy'))
    DIST_COEFFICIENTS = np.array([0., 0., 0., 0.]).reshape(4, 1)
    IMG_PATH = os.path.join(FILE_PATH, 'img.png')

    mat = aruco_detector(MARKER_LENGTH, CAMERA_INSTRINCS, DIST_COEFFICIENTS, IMG_PATH, print_flag=False, vis=False)
    T_camera_pose = np.array(mat[0])

    IMG_STORE_PATH = os.path.join(FILE_PATH, str(ID))
    if os.path.exists(IMG_STORE_PATH) == False:
        os.mkdir(IMG_STORE_PATH)
    cv2.imwrite(os.path.join(IMG_STORE_PATH, 'img.png'), img)

    if os.path.exists('configs/robot_calibration/') == False:
        os.makedirs('configs/robot_calibration/')

    with open(os.path.join('configs/robot_calibration/', '{}.npy'.format(ID)), 'wb') as fT:
        np.save(fT, T_camera_pose)

    if DEBUG:
        print('********* Calibration Times {} Finished *********'.format(ID))
        print('The pose matrix is ', T_camera_pose)


flexiv = FlexivRobot(robot_ip_address = robot_ip_address, pc_ip_address = pc_ip_address)

joint_poses = np.load('robot/robot_path/joint_poses.npy')

n_shape, _ = joint_poses.shape

flexiv.move_joint(joint_poses[0], duration = safe_duration)


for i in range(n_shape):
    joint_pos = joint_poses[i]
    print('[{}/{}] Move to --> {}'.format(i + 1, n_shape, joint_pos))
    if i == 1:
        flexiv.move_joint(joint_pos, duration = fast_duration * 2)
    else:
        flexiv.move_joint(joint_pos, duration = fast_duration)
    time.sleep(0.8)
    calib(i)
    

flexiv.move_joint(joint_poses[0], duration = safe_duration)