from ur_toolbox.camera import RealSense # pylint: disable=import-error
from multiprocessing import shared_memory
import numpy as np
from cv2 import cv2

camera = RealSense(frame_rate = 30, resolution = (1280, 720), resolution_depth = (1024, 768))

for i in range(10):
    colors, depths = camera.get_full_image()

shm_depth = shared_memory.SharedMemory(name='realsense_depth_L515', create=True, size=depths.nbytes)
depthbuf = np.ndarray(depths.shape, dtype=depths.dtype, buffer=shm_depth.buf)
colors = (cv2.cvtColor(colors, cv2.COLOR_BGR2RGB) / 255.0).astype(np.float32)
shm_color = shared_memory.SharedMemory(name='realsense_color_L515', create=True, size=colors.nbytes)
colorbuf = np.ndarray(colors.shape, dtype=colors.dtype, buffer=shm_color.buf)

try:
    while True:
        print('fetching images L515 ...')
        colors, depths = camera.get_full_image()
        colors = (cv2.cvtColor(colors, cv2.COLOR_BGR2RGB) / 255.0).astype(np.float32)

        colorbuf[:] = colors[:]
        depthbuf[:] = depths[:]
except KeyboardInterrupt:
    shm_depth.unlink()
    shm_color.unlink()