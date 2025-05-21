#!/usr/bin/env python3
import os, cv2, numpy as np, rospy, ros_numpy
from sensor_msgs.msg import PointCloud2

PIX_SHIFT = np.array([
     12, 12, 12, 12, 12, 12, 12, -4, 12, -4, 12, -4, 12, -4, 12, 4,
     -4,-12, 12, 4, -4,-12, 12, 4, -4,-12, 12, 4, -4,-12, 12, 4,
     -4,-12, 12, 4, -4,-12, 12, 4, -4,-12, 12, 4, -4,-12, 12, 4,
     -4,-12, 4,-12, 4,-12, 4,-12, 4,-12,-12,-12,-12,-12,-12,-12
], dtype=int)

import numpy as np
from typing import List

def destagger(pixel_shift_by_row: List[int], field: np.ndarray) -> np.ndarray:
    """Reference implementation for destaggering a field of data.

    In the default staggered representation, each column corresponds to a
    single timestamp. In the destaggered representation, each column
    corresponds to a single azimuth angle, compensating for the azimuth offset
    of each beam.

    Destaggering is used for visualizing lidar data as an image or for
    algorithms that exploit the structure of the lidar data, such as
    beam_uniformity in ouster_viz, or computer vision algorithms.

    Args:
        pixel_shift_by_row: List of pixel shifts by row from sensor metadata
        field: Staggered data as a H x W numpy array

    Returns:
        Destaggered data as a H x W numpy array
    """
    # Initialize output array with the same shape and dtype as the input
    destaggered = np.zeros(field.shape, dtype=field.dtype)
    nrows = field.shape[0]

    # iterate over every row and apply pixel shift
    for u in range(nrows):
        destaggered[u, :] = np.roll(field[u, :], pixel_shift_by_row[u])

    return destaggered


def to_mono8(img):
    img = cv2.normalize(img.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)
    return img.astype(np.uint8)

SAVE_ROOT = "/tmp/ouster_frames3"
os.makedirs(SAVE_ROOT, exist_ok=True)

def cloud_cb(msg):
    cloud = ros_numpy.numpify(msg).reshape(msg.height, msg.width)
    dst   = destagger(PIX_SHIFT, cloud)
    ts    = msg.header.stamp.to_nsec()
    cv2.imwrite(f"{SAVE_ROOT}/range_{ts}.png",        to_mono8(dst['range']))
    cv2.imwrite(f"{SAVE_ROOT}/intensity_{ts}.png",    to_mono8(dst['intensity']))
    cv2.imwrite(f"{SAVE_ROOT}/reflectivity_{ts}.png", to_mono8(dst['reflectivity']))
    cv2.imwrite(f"{SAVE_ROOT}/ambient_{ts}.png",      to_mono8(dst['ambient']))
    rospy.loginfo_once(f"First frame written to {SAVE_ROOT}")

if __name__ == "__main__":
    rospy.init_node("destagger_to_disk")
    rospy.Subscriber("/main/points", PointCloud2, cloud_cb, queue_size=4)
    rospy.spin()
