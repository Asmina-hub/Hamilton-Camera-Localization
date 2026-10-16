"""Hand-drawn masks of the car body (wings, wheels, nose) in each camera image, 1506 x 728 px.

Drawn from the mean image of the training run; the body does not move in the image.
"""
import cv2
import numpy as np

POLYS = {
    'cl': [[(0, 520), (375, 520), (375, 485), (1100, 490), (1110, 420), (1345, 425), (1385, 510), (1505, 495),
            (1505, 727), (0, 727)]],
    'cr': [[(0, 440), (205, 440), (210, 360), (445, 360), (455, 425), (1315, 485), (1505, 495), (1505, 727),
            (0, 727)],
           [(0, 0), (75, 0), (40, 455), (0, 455)]],
    'rr': [[(0, 520), (540, 535), (1145, 540), (1160, 440), (1390, 440), (1395, 345), (1505, 295), (1505, 727),
            (0, 727)]],
    'fr': [[(0, 455), (300, 420), (620, 420), (820, 465), (1140, 535), (1190, 410), (1505, 395), (1505, 727),
            (0, 727)]],
    'fl': [[(1355, 535), (1485, 535), (1485, 605), (1355, 605)],
           [(780, 727), (1100, 630), (1505, 595), (1505, 727)]],
    'r': [[(0, 700), (1505, 700), (1505, 727), (0, 727)]],
    'rl': [],
}
LEFT_BORDER_PX = 12       # every camera has a black column on the left


def body_masks(w=1506, h=728, grow=10):
    out = {}
    for c, polys in POLYS.items():
        m = np.zeros((h, w), np.uint8)
        for p in polys:
            cv2.fillPoly(m, [np.array(p, np.int32)], 1)
        m[:, :LEFT_BORDER_PX] = 1
        out[c] = cv2.dilate(m, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)) > 0
    return out
