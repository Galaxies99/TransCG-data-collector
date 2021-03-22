from ur_toolbox.camera import RealSense
from multiprocessing import shared_memory
import numpy as np
import cv2

DEBUG = True

camera = RealSense(frame_rate = 30)
if DEBUG:
    for i in range(10):
        colors, depths = camera.get_rgbd_image()
else:
    for i in range(10):
        depths = camera.get_depth_image()
        

shm_depth = shared_memory.SharedMemory(name='realsense_depth', create=True, size=depths.nbytes)
depthbuf = np.ndarray(depths.shape, dtype=depths.dtype, buffer=shm_depth.buf)

if DEBUG:
    colors = (cv2.cvtColor(colors, cv2.COLOR_BGR2RGB) / 255.0).astype(np.float32)
    shm_color = shared_memory.SharedMemory(name='realsense_color', create=True, size=colors.nbytes)
    colorbuf = np.ndarray(colors.shape, dtype=colors.dtype, buffer=shm_color.buf)

try:
    while True:
        colors = None
        
        if DEBUG:
            print(shm_color.name)
            print(shm_depth.name)
            colors, depths = camera.get_rgbd_image()
            colors = (cv2.cvtColor(colors, cv2.COLOR_BGR2RGB) / 255.0).astype(np.float32)

            
            colorbuf[:] = colors[:]
            depthbuf[:] = depths[:]
        else:
            print(shm_depth.name)
            depths = camera.get_depth_image()

            depthbuf[:] = depths[:]
except KeyboardInterrupt:
    shm_depth.unlink()
    shm_color.unlink()