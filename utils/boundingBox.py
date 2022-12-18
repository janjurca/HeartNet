import numpy as np


def getBoundingBox(mask):
    x = np.sum(np.sum(mask, 0), 0)
    y = np.sum(np.sum(mask, 1), 1)
    z = np.sum(np.sum(mask, 2), 0)
