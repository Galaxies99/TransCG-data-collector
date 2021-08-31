import os
import sys
import time
import numpy as np
from cv2 import cv2
from realsense import RealSense


class RealSenseCamera(object):
    def __init__(self, type='D435', use_infrared=False):
        self.type = type
        if type == 'D435':
            self.camera = RealSense(frame_rate = 30, use_infrared = use_infrared)
        if type == 'L515':
            self.camera = RealSense(frame_rate = 30, resolution = (1280, 720), resolution_depth = (1024, 768))
    
    def get_full_image(self):
        '''
        Function:
            Get the 720x1280x3 RGB image together with a 720x1280 depth image from the realsense camera.
            If 'use_infrared' is True, then return infrared images together.
        '''
        try:
            res = self.camera.get_full_image()
        except RuntimeError:
            res = self.camera.get_full_image()
        return res


if __name__ == '__main__':
    cam = RealSenseCamera()
    while True:
        colors, depths = cam.get_full_image()
        cv2.imshow('test', depths / 1000 * 255)
        cv2.waitKey(1)
    
    