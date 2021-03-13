import os
import sys
import time
from multiprocessing import shared_memory
import numpy as np
import cv2

class RealSenseCamera(object):
    def __init__(self):
        self.existing_shm_color = shared_memory.SharedMemory(name='realsense_color')
        self.existing_shm_depth = shared_memory.SharedMemory(name='realsense_depth')
    
    def get_rgbd(self):
        '''
        Function:
            Get the 720x1280x3 RGB image together with a 720x1280 depth image from the realsense camera. Need to execute "realsense.py" first.
        '''
        time.sleep(0.1)
        colors = np.copy(np.ndarray((720, 1280, 3), dtype=np.float32, buffer=self.existing_shm_color.buf))
        depths = np.copy(np.ndarray((720, 1280), dtype=np.uint16, buffer=self.existing_shm_depth.buf))
        return colors, depths


if __name__ == '__main__':
    cam = RealSenseCamera()
    colors, depths = cam.get_rgbd()
    cv2.imshow('test', colors[:, :, ::-1])
    cv2.waitKey(0)
    

