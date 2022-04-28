# This is a minimal example to work with poses and depth images, in case you want to use the datasets for another project.
# Just move this into the root directory of any of the synthetic datasets (such as cube/), where the intrinsics.txt file, rgb, pose and depth directories reside.

import numpy as np
import os
from glob import glob
import cv2

from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt


pose_fpaths = sorted(glob("pose/*.txt"))
depth_fpaths = sorted(glob("depth/*.png"))

intrinsics = np.genfromtxt("intrinsics.txt", max_rows=1)
f, cx, cy = intrinsics[:3]

# The intrinsics are for a sensor of shape (width, height) = (640, 480), where f is the vertical focal length.
cx *= 512./640
cy *= 512./480
f *= 512/480.

v, u = np.mgrid[:512,:512] # the first axis is image y-coordinates, since its the rows, the second axis is image x-coordinates (columns)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(5):
    pose = np.genfromtxt(pose_fpaths[i]).reshape(4,4)
    depth = cv2.imread(depth_fpaths[i], cv2.IMREAD_UNCHANGED).astype(np.float32) * 1e-4

    x_cam = depth * (u - cx) / f
    y_cam = depth * (v - cy) / f
    cam_coords = np.stack((x_cam, y_cam, depth, np.ones_like(depth)), axis=-1)

    cam_coords = cam_coords[depth!=1]

    world_coords = np.matmul(pose, cam_coords.reshape(-1, 4, 1)).squeeze()
    world_coords_subsampled = world_coords[np.random.randint(len(world_coords), size=1000)]

    x_wrld, y_wrld, z_wrld = np.split(world_coords_subsampled, 4, axis=1)[:3]

    ax.scatter(x_wrld,y_wrld, z_wrld)

plt.show()

