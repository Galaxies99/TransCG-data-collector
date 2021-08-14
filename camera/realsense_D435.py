from ur_toolbox.camera import RealSense # pylint: disable=import-error
from multiprocessing import shared_memory
import numpy as np
from cv2 import cv2

camera = RealSense(frame_rate = 30, use_infrared=True)
for i in range(10):
    colors, depths, infrared_left, infrared_right = camera.get_full_image()

shm_depth = shared_memory.SharedMemory(name='realsense_depth_D435', create=True, size=depths.nbytes)
depthbuf = np.ndarray(depths.shape, dtype=depths.dtype, buffer=shm_depth.buf)
colors = (cv2.cvtColor(colors, cv2.COLOR_BGR2RGB) / 255.0).astype(np.float32)
shm_color = shared_memory.SharedMemory(name='realsense_color_D435', create=True, size=colors.nbytes)
colorbuf = np.ndarray(colors.shape, dtype=colors.dtype, buffer=shm_color.buf)
shm_infrared_left = shared_memory.SharedMemory(name='realsense_infrared_left_D435', create=True, size=infrared_left.nbytes)
infraredleftbuf = np.ndarray(infrared_left.shape, dtype=infrared_left.dtype, buffer=shm_infrared_left.buf)
shm_infrared_right = shared_memory.SharedMemory(name='realsense_infrared_right_D435', create=True, size=infrared_right.nbytes)
infraredrightbuf = np.ndarray(infrared_right.shape, dtype=infrared_right.dtype, buffer=shm_infrared_right.buf)

try:
    while True:
        print('fetching images D435 ...')
        colors, depths, infrared_left, infrared_right = camera.get_full_image()
        colors = (cv2.cvtColor(colors, cv2.COLOR_BGR2RGB) / 255.0).astype(np.float32)
            
        colorbuf[:] = colors[:]
        depthbuf[:] = depths[:]
        infraredleftbuf[:] = infrared_left[:]
        infraredrightbuf[:] = infrared_right[:]
except KeyboardInterrupt:
    shm_depth.unlink()
    shm_color.unlink()
    shm_infrared_left.unlink()
    shm_infrared_right.unlink()