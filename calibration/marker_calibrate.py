"""
Marker calibrate example
Author: Haoshu Fang, Xiaoshan Lin
"""

import os
import sys
#from lib_py.base import fvr_setup
import cv2
from cv2 import aruco
import numpy as np
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
from averageQuaternions import averageQuaternions


def aruco_detector(marker_length,
                   camera_matrix,
                   dist_coeffs,
                   img_path,
                   print_flag,
                   vis=True):
    """
    Args:
        marker_length: The length of the edge of each marker in unit(mm)
        camera_matrix: 3*3 camera matrix constructed by parameter [fx, fy, cx, cy]
        dist_coeffs: [[0,0,0,0]]
        img_path: The location of the image 
        print_flag: Print the transformation matrices from the markers frame to the camera frame
    """

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_7X7_250)
    arucoParams = aruco.DetectorParameters_create()
    arucoParams.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
    arucoParams.cornerRefinementWinSize = 5
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(
        img_gray, aruco_dict, parameters=arucoParams)
    if ids != None:
        rvec, tvec, _objPoints = aruco.estimatePoseSingleMarkers(
            corners, marker_length, camera_matrix, dist_coeffs)
        matrices = [None] * 20

        index = 0
        for i, j in zip(tvec, rvec):
            i = i / 1000
            r = R.from_rotvec(j.reshape(3, ))
            rot_mtx = r.as_matrix()
            trf_mtx = np.hstack((rot_mtx, i.reshape(3, 1)))
            trf_mtx = np.vstack((trf_mtx, np.array([0, 0, 0, 1])))
            if print_flag:
                print(trf_mtx)
                print(ids)
                print(index)
            matrices[ids[index][0]] = trf_mtx

            index += 1
        if print_flag:
            for i in range(20):
                print("Transformation matrix T_%d:" % (i))
                print(matrices[i])

        if vis:
            draw_img = aruco.drawDetectedMarkers(img, corners, ids,
                                                 (0, 255, 0))
            for r, t in zip(rvec, tvec):
                draw_img = aruco.drawAxis(draw_img, camera_matrix, dist_coeffs,
                                          r, t, 100)
            cv2.imshow("Detected markers", draw_img)
            img_name = 'detected_image/' + img_path.strip('rgbpn/') + 'png'
            cv2.waitKey(0)
    else:
        print("Markers are not detected.\r")
        cv2.destroyAllWindows()
        return None
    return matrices


def get_average_transformation(filenames, camera_matrix, print_flag=True):
    dist_coeffs = np.array([0., 0., 0., 0.]).reshape(4, 1)
    quaternion = [[] for i in range(20)]
    transl = [[] for i in range(20) ]
    for img_path in filenames:
        print(img_path)
        transformation_matrix = aruco_detector(39.5, camera_matrix, dist_coeffs,
                                               img_path, print_flag, vis=False)
        if len(transformation_matrix) < 20: 
            continue
        for i in range(0, 20, 1):
            print(i)
            if transformation_matrix[i] is not None:
                matrix = np.dot(np.linalg.inv(transformation_matrix[5]),
                            transformation_matrix[i])
                if print_flag:
                    print(matrix)
                q = Quaternion(matrix=matrix[0:3, 0:3])
                quaternion[i].append([q[0], q[1], q[2], q[3]])
                transl[i].append([matrix[0][3], matrix[1][3], matrix[2][3]])
    quaternion = np.array(quaternion)
    transl = np.array(transl)
    result = []
    for i in range(20):
        if len(quaternion[i]) != 0:
            average_q = averageQuaternions(np.array(quaternion[i]))
            average_q = Quaternion(average_q[0], average_q[1], average_q[2],
                                   average_q[3])
            if print_flag:
                print(average_q)
            average_transl = np.sum(np.array(transl[i]), axis=0) / (np.array(transl[i]).shape[0])
            average_transformation = average_q.transformation_matrix
            average_transformation[0:3, 3] = average_transl
            if print_flag:
                print("++++++++++++++++++++++++++")
                print('transformation matrix from %dth frame to 5th frame is:' %
                      (i + 1))
                print(average_transformation)
            result.append(average_transformation)
        else:
            result.append(np.eye(4))
    return np.array(result)


if __name__ == '__main__':

    filename = os.listdir('../scenes/scene_201_cali/data/rgb')
    filenames = [
        os.path.join('../scenes/scene_201_cali/data/rgb', name)
        for name in filename
    ]
    camera_matrix = np.load('../scenes/scene_201_cali/data/camK.npy')
    a = get_average_transformation(filenames, camera_matrix)
    np.save("markers_transformation_cali201.npy", a)
