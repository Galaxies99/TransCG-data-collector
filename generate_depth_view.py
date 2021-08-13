import cv2
import os
from tqdm import tqdm

for i in tqdm(range(1, 111)):
    for j in range(240):
        p = os.path.join('data', 'scene{}'.format(i), str(j))
        # a = cv2.imread(os.path.join(p, 'depth1.png'), cv2.IMREAD_UNCHANGED)
        # a = a / 1000 * 255
        # cv2.imwrite(os.path.join(p, 'depth1-view.png'), a)
        b = cv2.imread(os.path.join(p, 'depth2.png'), cv2.IMREAD_UNCHANGED)
        b = b / 3000 * 255
        cv2.imwrite(os.path.join(p, 'depth2-view.png'), b)
