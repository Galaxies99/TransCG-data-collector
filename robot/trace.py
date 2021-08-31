import numpy as np
import time
import os
from flexiv import FlexivRobot
import numpy as np
from cv2 import cv2
import netrelay.client as client
from camera.camera import RealSenseCamera
from utils.pst_jsonhandler import formatter_str, find_obj


robot_ip_address = '192.168.2.100'
pc_ip_address = '192.168.2.200'
safe_duration = 5
normal_duration = 3
fast_duration = 1


camera1 = RealSenseCamera(type='D435')
camera2 = RealSenseCamera(type='L515')
s, _ = client.start(('10.52.25.177', 23333))

object_file_name_list_file = open('object_file_name_list.txt', 'r')
lines = object_file_name_list_file.readlines()
objectfilenamelist = []
for line in lines:
    if not (line == '\n'):
        objectfilenamelist.append(line.replace('\n','').replace('\r',''))

last_image_depth = np.zeros((720, 1280))
last_image_depth2 = np.zeros((720, 1280))

def fetch_data(ID, TIME):
    global last_image_depth, last_image_depth2
    pose = []

    DATA_DIR = os.path.join('data', 'scene{}'.format(ID))
    CUR_DATA_DIR = os.path.join(DATA_DIR, str(TIME))
    if os.path.exists(CUR_DATA_DIR) == False:
        os.makedirs(CUR_DATA_DIR)
    
    obj_id_list = range(len(objectfilenamelist))
    
    pose = [None] * len(obj_id_list)
    
    T_tracker_camera = np.load('configs/T_tracker_camera.npy')
    
    cmd_tracker = 'GetTracker'
    
    image, image_depth = camera1.get_full_image()
    image2, image_depth2 = camera2.get_full_image()
    tracker_res = client.exec_cmd(s, cmd_tracker)
    tracker_js = formatter_str(tracker_res)

    for i, obj_id in enumerate(obj_id_list):
        notfound = False
        obj_filename = objectfilenamelist[obj_id]
        obj_name, _ = os.path.splitext(obj_filename)
        try:
            T_tracker_marker = find_obj(tracker_js, obj_name)
        except ValueError:
            notfound = True
            
        transformation_file = 'results/transformation/{}.npy'.format(obj_id)
        if not os.path.exists(transformation_file):
            notfound = True
        else:
            T_marker_object = np.load(transformation_file)
    
        if not notfound:
            T_camera_object = (np.linalg.inv(T_tracker_camera).dot(T_tracker_marker)).dot(T_marker_object)
            pose[i] = T_camera_object
        else:
            pose[i] = None
    
    POSE_DIR = os.path.join(CUR_DATA_DIR, 'pose')
    if os.path.exists(POSE_DIR) == False:
        os.makedirs(POSE_DIR)
    cv2.imwrite(os.path.join(CUR_DATA_DIR, 'rgb1.png'), image)
    cv2.imwrite(os.path.join(CUR_DATA_DIR, 'rgb2.png'), image2)
    cv2.imwrite(os.path.join(CUR_DATA_DIR, 'depth1.png'), image_depth)
    image_depth_ = image_depth / 1000 * 255
    cv2.imwrite(os.path.join(CUR_DATA_DIR, 'depth1-view.png'), image_depth_)
    cv2.imwrite(os.path.join(CUR_DATA_DIR, 'depth2.png'), image_depth2)
    image_depth2_ = image_depth2 / 3000 * 255
    cv2.imwrite(os.path.join(CUR_DATA_DIR, 'depth2-view.png'), image_depth2_)
    for i, p in enumerate(pose):
        if p is None:
            continue
        else:
            np.save(os.path.join(POSE_DIR, '{}.npy'.format(i)), p)


flexiv = FlexivRobot(robot_ip_address = robot_ip_address, pc_ip_address = pc_ip_address)

joint_poses = np.load('robot/robot_path/joint_poses.npy')

n_shape, _ = joint_poses.shape

flexiv.move_joint(joint_poses[0], duration = safe_duration)

scene_id = int(input('Scene ID'))

for i in range(n_shape):
    joint_pos = joint_poses[i]
    print('[{}/{}] Move to --> {}'.format(i + 1, n_shape, joint_pos))
    if i == 1:
        flexiv.move_joint(joint_pos, duration = fast_duration * 2)
    else:
        flexiv.move_joint(joint_pos, duration = fast_duration)
    time.sleep(0.8)
    fetch_data(scene_id, i)

flexiv.move_joint(joint_poses[0], duration = safe_duration)

client.close(s)