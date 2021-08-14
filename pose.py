"""
Utils Functions for Pose Transformation
Author: Chongzhao Mao, Lingxiao Song
"""

import numpy as np
from transforms3d.euler import quat2euler
from pyquaternion import Quaternion
import cv2

def compute_norm(v):
    """
    Compute norm of a vector.
    
    Args:
        v: numpy array of arbitrary length.
    Returns:
        norm: float, norm of vector.
    """
    norm = np.linalg.norm(v)
    return norm


def quat_to_euler(quat):
    """
    Convert quaternion to euler representation.

    Args:
        quat: 4D numpy array.
    Returns: 
        euler: 3D numpy array.
    """
    euler = np.array(quat2euler(quat))
    return euler


def quat_to_degree(quat):
    """
    Convert quaternion to degree representation.

    Args:
        quat: 4D numpy array.
    Returns: 
        degree: 3D numpy array.
    """
    return quat_to_euler(quat) / np.pi * 180


def compute_angle_two_quat(quat_0, quat_1, out_dim=1):
    """
    Compute angle in degree of two quaternions.

    Args:
        quat_0: 4D numpy array.
        quat_1: 4D numpy array.
        out_dim: (optional) 1 or 3. 1 for norm of 3 dimension, 3 for each dimension. 
    Returns:
        degree_diff: 1D or 3D numpy array.
    """
    degree_0 = quat_to_degree(quat_0)
    degree_1 = quat_to_degree(quat_1)
    degree_diff = np.abs(degree_0 - degree_1)
    degree_diff = np.minimum(degree_diff, 360 - degree_diff)
    if out_dim == 1:
        degree_diff = compute_norm(degree_diff)
    elif out_dim == 3:
        pass
    return degree_diff

