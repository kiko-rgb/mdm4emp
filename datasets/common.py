from collections import Counter
from pathlib import Path
import cv2
import numpy as np

import torch


def load_image(path: Path) -> torch.Tensor:
    """Load an image from a file.

    Args:
        path (Path): path to the image file

    Returns:
        torch.Tensor: tensor of shape (H, W, C) containing the image
    """
    return torch.from_numpy(
        cv2.cvtColor(
            cv2.imread(path.as_posix()),
            cv2.COLOR_BGR2RGB,
        ).astype(np.float32)
        / 255
    )


def load_lidar_points(path: Path) -> torch.Tensor:
    """Load lidar points from a binary file.

    Args:
        path (Path): path to the binary file

    Returns:
        torch.Tensor: tensor of shape (N, 4) containing the lidar points, xyz and intensity
    """
    return torch.from_numpy(np.fromfile(path, dtype=np.float32).reshape(-1, 4))


def lidar_to_depth_map(
    points: torch.Tensor,
    K: torch.Tensor,
    image_shape: tuple[int, int],
    lidar2cam_pose: torch.Tensor,
) -> torch.Tensor:
    """Project lidar points into an image.

    Args:
        points (torch.Tensor): tensor of shape (N, 4) containing the lidar points, xyz and intensity
        K (torch.Tensor): camera intrinsics
        image_shape (tuple[int, int]): image shape
        T (torch.Tensor): translation
        R (torch.Tensor): rotation

    Returns:
        torch.Tensor: depth map (H, W)
    """
    # project points to camera
    points = points[:, :3]
    points = points @ lidar2cam_pose[:3, :3].T + lidar2cam_pose[:3, 3]
    points = points @ K.T
    proj_points = points / points[:, 2, None]

    # remove points outside the image and points behind the camera
    mask = (
        (proj_points[:, 0] >= 0)
        & (proj_points[:, 0] < image_shape[1])
        & (proj_points[:, 1] >= 0)
        & (proj_points[:, 1] < image_shape[0])
        & (points[:, 2] > 0)
    )
    proj_points = proj_points[mask]
    points = points[mask]

    # create depth map
    depth_map = torch.zeros(image_shape)
    depth_map[proj_points[:, 1].long(), proj_points[:, 0].long()] = points[:, 2]

    inds = proj_points[:, 1] * (image_shape[1] - 1) + proj_points[:, 0] - 1
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]

    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        x_loc = int(proj_points[pts[0], 0])
        y_loc = int(proj_points[pts[0], 1])
        depth_map[y_loc, x_loc] = points[pts, 2].min()

    return depth_map
