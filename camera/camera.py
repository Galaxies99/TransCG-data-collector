import os
import sys
import time
from multiprocessing import shared_memory
import numpy as np
from cv2 import cv2

class RealSenseCamera(object):
    def __init__(self, type='D435', resolution=(1280, 720), use_infrared = False):
        self.resolution = resolution
        self.existing_shm_color = shared_memory.SharedMemory(name='realsense_color_{}'.format(type))
        self.existing_shm_depth = shared_memory.SharedMemory(name='realsense_depth_{}'.format(type))
        self.use_infrared = use_infrared
        if use_infrared:
            self.existing_shm_infrared_left = shared_memory.SharedMemory(name='realsense_infrared_left_{}'.format(type))
            self.existing_shm_infrared_right = shared_memory.SharedMemory(name='realsense_infrared_right_{}'.format(type))
    
    def get_full_image(self):
        '''
        Function:
            Get the 720x1280x3 RGB image together with a 720x1280 depth image from the realsense camera.
            If 'use_infrared' is True, then return infrared images together.
        '''
        time.sleep(0.1)
        colors = np.copy(np.ndarray((self.resolution[1], self.resolution[0], 3), dtype=np.float32, buffer=self.existing_shm_color.buf))
        depths = np.copy(np.ndarray((self.resolution[1], self.resolution[0]), dtype=np.uint16, buffer=self.existing_shm_depth.buf))
        colors = (colors * 255).astype(np.uint8)
        colors = cv2.cvtColor(colors, cv2.COLOR_RGB2BGR)
        if self.use_infrared:
            infrared_left = np.copy(np.ndarray((self.resolution[1], self.resolution[0]), dtype=np.uint8, buffer=self.existing_shm_infrared_left.buf))
            infrared_right = np.copy(np.ndarray((self.resolution[1], self.resolution[0]), dtype=np.uint8, buffer=self.existing_shm_infrared_right.buf))
            return colors, depths, infrared_left, infrared_right
        else:
            return colors, depths
