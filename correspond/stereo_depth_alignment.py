import torch
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from LightGlue.lightglue import LightGlue, SuperPoint
from LightGlue.lightglue.utils import load_image, rbd

def matching(device: str = "cuda"):
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
    matcher = LightGlue(features='superpoint').eval().to(device)
    
    def _match(image0: torch.Tensor, image1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats0 = extractor.extract(image0)
        feats1 = extractor.extract(image1)


        matches01 = matcher({'image0': feats0, 'image1': feats1})
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
        matches = matches01['matches']  # indices with shape (K,2)


        points0 = feats0['keypoints'][matches[..., 0]].cpu().numpy()  # coordinates in image #0, shape (K,2)
        points1 = feats1['keypoints'][matches[..., 1]].cpu().numpy()  # coordinates in image #1, shape (K,2)

        return points0, points1


    return _match

def filter_keypoints_epipolar_error(points0: torch.Tensor, points1: torch.Tensor, rel_pose: torch.Tensor, projs: torch.Tensor):
    
    T = rel_pose[:3, 3]
    T_x = torch.torch([
        [0, -T[2], T[1]],
        [T[2], 0, -T[0]],
        [-T[1], T[0], 0]
    ])

    E = T_x @ rel_pose[:3, :3] # (3, 3)

    points0_normalized = torch.linalg.inv(projs[0]) @ torch.nn.functional.pad(points0, (0, 1), value=1).permute(1, 0) # (3, N)
    points1_normalized = torch.linalg.inv(projs[1]) @ torch.nn.functional.pad(points1, (0, 1), value=1).permute(1, 0) # (3, N)

    epipolar_error = torch.einsum("i...,ij,...j->...", points0_normalized, E, points1_normalized) # (N)
    epipolar_errors = []
    for i in range(points0_normalized.shape[1]):
        p1 = points0_normalized[:, i]
        p2 = points1_normalized[:, i]
        result = p2.T @ E @ p1
        epipolar_errors.append(result)

    mean = torch.mean(epipolar_errors)
    threshold = mean
    # filter
    epipolar_distances = torch.abs(epipolar_error)
    mask = epipolar_distances < threshold

    
    filtered_points0 = points0[mask]
    filtered_points1 = points1[mask]
    
    points0 = torch.tensor([[100, 150], [200, 250], [300, 350]], dtype=torch.float32)
    points1 = torch.tensor([[110, 160], [210, 260], [310, 360]], dtype=torch.float32)
    rel_pose = torch.eye(4)
    rel_pose[:3, 3] = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
    projs = torch.stack([torch.eye(3), torch.eye(3)])

    return points0[mask], points1[mask]

def optimize_scale_and_shift(points0_normalized: torch.Tensor, points1_normalized: torch.Tensor, depths0: torch.Tensor, depths1: torch.Tensor, rel_pose: torch.Tensor) -> torch.Tensor:
    # points0_normalized (N, 3) K^-1@p
    # points1_normalized (N, 3) K^-1@p
    # depths0 (N)
    # depths1 (N)
    rotated_points1 = rel_pose[None, :3,:3] @ points1_normalized

    p0_times_d = points0_normalized * depths0.unsqeeze(-1)
    p1_times_d = rotated_points1 * depths1.unsqeeze(-1)

    W = torch.concat([p0_times_d.flatten(0, 1), points0_normalized.flatten(0, 1), -p1_times_d.flatten(0, 1), -rotated_points1.flatten(0, 1)], dim=-1) # (N*3,4)

    Y = rel_pose[:3, 3:4].repeat(points0_normalized.shape[0]) # (N*3)

    solution = torch.linalg.inv(W.permute(1, 0) @ W) @ W.permute(1, 0) @ Y

    return solution

def extract_depth(depth_map: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    depths = []
    for point in points:
        depths.append(depth_map[point[1], point[0]])

    return depths


def stereo_depth_alignment(images: torch.Tensor, depth_maps: torch.Tensor, projs: torch.Tensor, poses: torch.Tensor, matching_fn):
    """
    images (2, 3, H, W)
    depthmaps (2, H, W)
    projs (2, 3, 3)
    poses (2, 4, 4)
    """
    rel_pose = torch.linalg.inv(poses[0]) @ poses[1] # image 1 to image 0

    points0, points1 = matching_fn(images[0], images[1])

    points0, points1 = filter_keypoints_epipolar_error(points0, points1, rel_pose, projs)

    depth0 = extract_depth(depth_maps[0], points0)
    depth1 = extract_depth(depth_maps[1], points0)

    points0_normalized = torch.linalg.inv(projs[0]) @ torch.nn.functional.pad(points0, (0, 1), value=1).permute(1, 0) # (3, N)
    points1_normalized = torch.linalg.inv(projs[1]) @ torch.nn.functional.pad(points1, (0, 1), value=1).permute(1, 0) # (3, N)

    parameters = optimize_scale_and_shift(points0_normalized, points1_normalized, depth0, depth1, rel_pose)

    depth0 = parameters[0] * depth_maps[0] + parameters[1]
    depth1 = parameters[2] * depth_maps[1] + parameters[3]

    return torch.stack([depth0, depth1], dim=0), parameters

# NOTE!!! the dataloader returns a normalized proj matrix. Talk to miguel about this.