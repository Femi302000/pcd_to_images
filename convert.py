#!/usr/bin/env python3
import os
import argparse
import numpy as np
import rosbag
import ros_numpy
import cv2
import matplotlib.pyplot as plt

def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def read_calibration(bag, info_topic):
    for _, msg, _ in bag.read_messages(topics=[info_topic]):
        alt   = np.array(msg.beam_altitude_angles, dtype=np.float64)
        azi   = np.array(msg.beam_azimuth_angles, dtype=np.float64)
        shift = np.array(msg.pixel_shift_by_row, dtype=np.int32)
        return np.deg2rad(alt), np.deg2rad(azi), shift
    raise RuntimeError(f"No messages on topic {info_topic}")

def build_images_from_msg(msg, alt_rad, pixel_shift, HORIZ_RES=1024):
    """
    Given one PointCloud2 msg, returns dict of 64×1024 float32 arrays:
      - 'range'
      - 'intensity'
      - 'reflectivity'
      - 'signal'
    """
    # Load structured array of shape (64,1024)
    pc = ros_numpy.point_cloud2.pointcloud2_to_array(msg)

    # Flatten into 1D structured array of length 64*1024
    pc_flat = pc.reshape(-1)

    # Extract and cast channels
    x = pc_flat['x'].astype(np.float32)
    y = pc_flat['y'].astype(np.float32)
    z = pc_flat['z'].astype(np.float32)
    intensity = (pc_flat['intensity'].astype(np.float32)
                 if 'intensity' in pc_flat.dtype.names else np.zeros_like(x))
    reflectivity = (pc_flat['reflectivity'].astype(np.float32)
                    if 'reflectivity' in pc_flat.dtype.names else np.zeros_like(x))
    signal = (pc_flat['signal'].astype(np.float32)
              if 'signal' in pc_flat.dtype.names else np.zeros_like(x))

    # Compute per-point range and mask out zeros
    r = np.linalg.norm(np.vstack((x, y, z)).T, axis=1)
    valid = r > 0

    x = x[valid];   y = y[valid];   z = z[valid]
    r = r[valid]
    intensity   = intensity[valid]
    reflectivity= reflectivity[valid]
    signal      = signal[valid]

    # Compute spherical coordinates
    phi   = np.arcsin(z / r)                          # elevation
    theta = np.mod(np.arctan2(y, x), 2*np.pi)         # azimuth [0,2π)

    # Determine beam row and column index
    beam_idx = np.argmin(np.abs(phi[:,None] - alt_rad[None,:]), axis=1)
    col = ((theta / (2*np.pi) * HORIZ_RES).astype(int) % HORIZ_RES)
    col = (col + pixel_shift[beam_idx]) % HORIZ_RES

    VERT_RES = len(alt_rad)
    # Initialize output images
    imgs = {
        'range'       : np.full((VERT_RES, HORIZ_RES), np.nan, dtype=np.float32),
        'intensity'   : np.full((VERT_RES, HORIZ_RES), np.nan, dtype=np.float32),
        'reflectivity': np.full((VERT_RES, HORIZ_RES), np.nan, dtype=np.float32),
        'signal'      : np.full((VERT_RES, HORIZ_RES), np.nan, dtype=np.float32),
    }

    # Fill each bin with the closest return
    for i in range(len(r)):
        row, c = beam_idx[i], col[i]
        prev_r = imgs['range'][row, c]
        if np.isnan(prev_r) or r[i] < prev_r:
            imgs['range'][row, c]        = r[i]
            imgs['intensity'][row, c]    = intensity[i]
            imgs['reflectivity'][row, c] = reflectivity[i]
            imgs['signal'][row, c]       = signal[i]

    # Also return raw xyz for scatter plotting
    return imgs, (x, y, z)

def save_channel(arr, prefix, name):
    """
    Save one channel to:
      - {prefix}_{name}.npy        (full precision)
      - {prefix}_{name}.png        (16-bit normalized)
      - {prefix}_{name}_viz.png    (8-bit preview)
    """
    # Save raw float32
    np.save(f"{prefix}_{name}.npy", arr)

    # 16-bit PNG (per-frame normalization)
    finite = arr[np.isfinite(arr)]
    if finite.size:
        mn, mx = finite.min(), finite.max()
        scaled = ((arr - mn) / max(mx-mn, 1e-6) * 65535).astype(np.uint16)
    else:
        scaled = np.zeros_like(arr, dtype=np.uint16)
    cv2.imwrite(f"{prefix}_{name}.png", scaled)

    # 8-bit preview
    plt.imsave(f"{prefix}_{name}_viz.png", arr,
               cmap='gray', vmin=0, vmax=np.nanmax(arr))

def save_pointcloud_image(x, y, z, prefix):
    """
    Save a top-down (bird's-eye) scatter plot of the point cloud.
    """
    plt.figure(figsize=(6,6))
    plt.scatter(x, y, c=z, s=1, cmap='viridis', marker='.')
    plt.colorbar(label='Z (m)')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('LiDAR Points (Bird\'s-eye view)')
    plt.axis('equal')
    plt.tight_layout()
    out_png = f"{prefix}_points.png"
    plt.savefig(out_png, dpi=200)
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract range/intensity/reflectivity/signal images plus raw point cloud from an Ouster bag")
    parser.add_argument('--bagfile',     required=True, help="Path to your .bag file")
    parser.add_argument('--output_dir',  required=True, help="Directory where outputs will be saved")
    parser.add_argument('--info_topic',   default='/main/ouster_info', help="Calibration info topic")
    parser.add_argument('--points_topic', default='/main/points',      help="PointCloud2 topic")
    parser.add_argument('--single_frame', action='store_true',
                        help="If set, only process the first scan and exit")
    args = parser.parse_args()

    # Read calibration once
    bag = rosbag.Bag(args.bagfile, 'r')
    alt_rad, azi_off_rad, pixel_shift = read_calibration(bag, args.info_topic)

    # Prepare output folder
    base_name = os.path.splitext(os.path.basename(args.bagfile))[0]
    out_folder = os.path.join(args.output_dir, base_name)
    make_dirs(out_folder)

    # Process each PointCloud2 message
    for idx, (_, msg, _) in enumerate(bag.read_messages(topics=[args.points_topic])):
        imgs, (x, y, z) = build_images_from_msg(msg, alt_rad, pixel_shift)
        prefix = os.path.join(out_folder, f'frame_{idx:04d}')

        # save each LiDAR image channel
        for ch in ['range', 'intensity', 'reflectivity', 'signal']:
            save_channel(imgs[ch], prefix, ch)

        # save the raw point cloud scatter plot
        save_pointcloud_image(x, y, z, prefix)

        print(f"[+] Saved frame {idx} → {prefix}_*.png")
        if args.single_frame:
            break

    bag.close()
    print("All done. Outputs saved under:", out_folder)
