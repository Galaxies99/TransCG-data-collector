import os
import sys
import time
from multiprocessing import shared_memory
import numpy as np
from cv2 import cv2

class RealSenseCamera(object):
    def __init__(self, type='D435', resolution=(1280, 720)):
        self.resolution = resolution
        self.existing_shm_color = shared_memory.SharedMemory(name='realsense_color_{}'.format(type))
        self.existing_shm_depth = shared_memory.SharedMemory(name='realsense_depth_{}'.format(type))
    
    def get_rgbd(self):
        '''
        Function:
            Get the 720x1280x3 RGB image together with a 720x1280 depth image from the realsense camera. Need to execute "realsense.py" first.
        '''
        time.sleep(0.1)
        colors = np.copy(np.ndarray((self.resolution[1], self.resolution[0], 3), dtype=np.float32, buffer=self.existing_shm_color.buf))
        depths = np.copy(np.ndarray((self.resolution[1], self.resolution[0]), dtype=np.uint16, buffer=self.existing_shm_depth.buf))
        return colors, depths


if __name__ == '__main__':
    cam = RealSenseCamera()
    while True:
        colors, depths = cam.get_rgbd()
        cv2.imshow('test', colors[:, :, ::-1])
        cv2.waitKey(1)
    

