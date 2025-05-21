#!/usr/bin/env python3
"""Destagger Ouster point clouds and write color‑mapped range/intensity/reflectivity/ambient images.

This version fetches the per‑row pixel shifts dynamically from the
`/main/ouster_info` topic (`OusterSensorInfo` message) instead of
using a hard‑coded table.
"""

from pathlib import Path

import cv2
import numpy as np
import rospy
import ros_numpy
from sensor_msgs.msg import PointCloud2

from ouster_ros.msg import OusterSensorInfo

from typing import Optional

PIX_SHIFT: Optional[np.ndarray] = None


def destagger(pixel_shift_by_row: np.ndarray, field: np.ndarray) -> np.ndarray:
    """Apply per‑row circular shifts to a 2‑D LiDAR field."""
    dest = np.empty_like(field)
    for row, shift in enumerate(pixel_shift_by_row):
        dest[row] = np.roll(field[row], shift)
    return dest


def to_mono8(img: np.ndarray) -> np.ndarray:
    """Normalize image to 0‑255 and cast to uint8."""
    norm = cv2.normalize(img.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)
    return norm.astype(np.uint8)


def to_colormap(img: np.ndarray, cmap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """Convert a single‑channel image to a color‑mapped BGR image."""
    return cv2.applyColorMap(to_mono8(img), cmap)


OUT_DIR = Path("/tmp/ouster_images_final")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def ouster_info_cb(msg: OusterSensorInfo) -> None:
    """Cache `pixel_shift_by_row` from the sensor‑info message."""
    global PIX_SHIFT
    PIX_SHIFT = np.asarray(msg.pixel_shift_by_row, dtype=int)
    rospy.loginfo_once(
        "Received OusterSensorInfo; pixel‑shift table loaded (len=%d)", len(PIX_SHIFT)
    )


def cloud_cb(msg: PointCloud2) -> None:
    """Process each incoming point cloud and write color‑mapped images to disk."""
    if PIX_SHIFT is None:
        rospy.logwarn_throttle(10.0, "Waiting for /main/ouster_info; skipping clouds…")
        return

    cloud = ros_numpy.numpify(msg).reshape(msg.height, msg.width)
    dst = destagger(PIX_SHIFT, cloud)
    ts = msg.header.stamp.to_nsec()

    channels = {
        "range": (dst["range"], cv2.COLORMAP_JET),
        "intensity": (dst["intensity"], cv2.COLORMAP_HOT),
        "reflectivity": (dst["reflectivity"], cv2.COLORMAP_PLASMA),
        "ambient": (dst["ambient"], cv2.COLORMAP_VIRIDIS),
    }

    for name, (field, cmap) in channels.items():
        cv2.imwrite(str(OUT_DIR / f"{name}_{ts}.png"), to_colormap(field, cmap))

    rospy.loginfo_once("First colored frame written to %s", OUT_DIR)


def main() -> None:
    rospy.init_node("destagger_to_disk_colormap")

    rospy.Subscriber("/main/ouster_info", OusterSensorInfo, ouster_info_cb, queue_size=1)
    rospy.Subscriber("/main/points", PointCloud2, cloud_cb, queue_size=4)

    rospy.loginfo("destagger_to_disk_colormap node started. Waiting for data …")
    rospy.spin()


if __name__ == "__main__":
    main()
