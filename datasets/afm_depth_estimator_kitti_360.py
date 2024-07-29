import os
import numpy as np
import torch
from datasets.afm_kitti_360 import AFMKITTI360Dataset
from correspond.stereo_depth_alignment import stereo_depth_alignment, matching
from PIL import Image
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
print(module_path, sys.path)
from lang_segment_anything import LangSAM

# Save point cloud to obj file
def save_point_cloud_to_obj(points, filename, colors=None):
    with open(filename  + '.obj', 'w') as f:
        for i in range(points.shape[0]):
            # Write the vertices with colors
            point = points[i, :]
            if colors is not None:
                color = colors[i, :]
                f.write('v %f %f %f %f %f %f\n' % (point[0], point[1], point[2], color[0], color[1], color[2]))
            else:
                f.write('v %f %f %f\n' % (point[0], point[1], point[2]))

def align_depth_maps(observation: np.ndarray, target: np.ndarray, weights: np.ndarray) -> np.ndarray:
    A = torch.nn.functional.pad(torch.from_numpy(observation).float().unsqueeze(-1), (0, 1), value=1.0) # N, 2
    b = torch.from_numpy(target).float()
    W = torch.diag(torch.from_numpy(weights).float())

    parameters = torch.linalg.inv(A.T @ W @ A) @ A.T @ W @ b

    return parameters.numpy()

def get_mask(masks_path, sequence, index, image_shape):
     sample_mask = str(index).zfill(10) + '.npz'
     mask_path = os.path.join(masks_path, sequence, sample_mask)
     # Check if file exists
     if not os.path.exists(mask_path):
        print(image_shape)
        return np.ones([image_shape[0], image_shape[1]], dtype=bool)
     else:
        return np.load(mask_path)['x']

def edge_filter(
    depth_map: torch.Tensor, threshold: float, neighborhood: str = "4"
) -> torch.Tensor:
    def _4_neighborhood(depth_map: torch.Tensor) -> torch.Tensor:
        depth_map = torch.stack(
            [
                depth_map[1:-1, 1:-1],
                depth_map[1:-1, :-2],
                depth_map[1:-1, 2:],
                depth_map[:-2, 1:-1],
                depth_map[2:, 1:-1],
            ],
            dim=0,
        )
        return depth_map

    def _8_neighborhood(depth_map: torch.Tensor) -> torch.Tensor:
        depth_map = torch.stack(
            [
                depth_map[1:-1, 1:-1],
                depth_map[1:-1, :-2],
                depth_map[1:-1, 2:],
                depth_map[:-2, 1:-1],
                depth_map[2:, 1:-1],
                depth_map[:-2, :-2],
                depth_map[2:, 2:],
                depth_map[:-2, 2:],
                depth_map[2:, :-2],
            ],
            dim=0,
        )
        return depth_map

    match neighborhood:
        case "4":
            depth_neighbors = _4_neighborhood(depth_map)
        case "8":
            depth_neighbors = _8_neighborhood(depth_map)
        case _:
            raise ValueError("Unknown neighborhood")

    depth_difference = torch.abs(depth_neighbors[1:] - depth_neighbors[0])
    mask = torch.all(depth_difference < threshold, dim=0)
    mask = torch.nn.functional.pad(mask, (1, 1, 1, 1), value=False)

    return mask

class AFMDepthEstimatorKITTI360Dataset(AFMKITTI360Dataset):
    def __init__(self, depth_estimator_model, estimator_mode, use_sky_masks=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.depth_estimator_model = depth_estimator_model
        self.estimator_mode = estimator_mode
        self.use_sky_masks = use_sky_masks
        self.masks_path = '/usr/prakt/s0081/AFM_project/mdmemp/masks_sky_lsa/'

        if self.use_sky_masks:
            # load LANG-Segment Anything model
            self.lang_sam_model = LangSAM(sam_type="vit_l")
            self.text_prompt = "sky"

        # K_perspective is in image coordinates [-1, 1], transform to pixel coordinates
        self.K = self._calibs['K_perspective'].copy()
        image_height, image_width = self.target_image_size[0], self.target_image_size[1]
        self.K[0, 0] = (self.K[0, 0] / 2.0) * image_width
        self.K[1, 1] = (self.K[1, 1] / 2.0) * image_height
        self.K[0, 2] = ((self.K[0, 2] + 1.0) / 2.0) * image_width
        self.K[1, 2] =  ((self.K[1, 2] + 1.0) / 2.0) * image_height
        # Get transformation matrix from camera to velodyne
        self.camera_to_velodyne = self._calibs['T_pose_to_velo'].dot(self._calibs['T_cam_to_pose']['00'])

    def __getitem__(self, idx):
        data = super().__getitem__(idx)

        # Predict the depth/disparity map
        left_image = data['imgs'][0]

        with torch.no_grad():
            pred = self.depth_estimator_model.infer_image(left_image) # HxW raw depth map
            print(pred.max())
        # Estimate the metric depth from the images
        if self.estimator_mode == 'metric_depth':
            metric_estimation = pred
        elif self.estimator_mode == 'relative_depth_aligned_to_lidar':
            # Compute the 3D points from the depth map
            depth_map = np.array(data['depths']).reshape(left_image.shape[0], left_image.shape[1])
            predicted_disparity = pred
            valid = depth_map > 0
            valid_ = valid & (predicted_disparity > 0.01)
            inverse = False
            if inverse:
                parameters = align_depth_maps(predicted_disparity[valid_], 1 / depth_map[valid_], 1 / depth_map[valid_])
                print('Scale: ', parameters[0], 'Shift: ', parameters[1])
                metric_estimation = 1.0 / (parameters[0] * predicted_disparity + parameters[1])
            else:
                parameters = align_depth_maps(1 / predicted_disparity[valid_], depth_map[valid_], predicted_disparity[valid_])
                print('Scale: ', parameters[0], 'Shift: ', parameters[1])
                # metric_estimation = np.clip(parameters[0] * (1 / predicted_disparity) + parameters[1], 0.0, 50.0)
                metric_estimation = parameters[0] * (1 / predicted_disparity) + parameters[1]
        elif self.estimator_mode == 'stereo_matching_alignment':
            right_image = data['imgs'][1]
            with torch.no_grad():
                right_pred = self.depth_estimator_model.infer_image(right_image) # HxW raw depth map
            # Perform stereo alignment
            images = torch.stack([left_image, right_image], dim=0)
            depth_maps = torch.stack([pred, right_pred], dim=0)
            projs = torch.stack([torch.tensor(self.K), torch.tensor(self.K)], dim=0)
            left_pose, right_pose = np.linalg.inv(data._calibs['T_cam_to_pose']['00']), np.linalg.inv(data._calibs['T_cam_to_pose']['01'])
            poses = torch.stack([torch.tensor(left_pose), torch.tensor(right_pose)], dim=0)
            depths, parameters = stereo_depth_alignment(images, depth_maps, projs, poses, matching)
            print('Scale: ', parameters[0], 'Shift: ', parameters[1])
            metric_estimation = np.clip(depths[0].numpy(), 0.0, 50.0)

        if self.use_sky_masks:
            # convert to PIL image
            image_pil = Image.fromarray((left_image*255).astype(np.uint8))
            # get mask from LANG_SAM
            masks, _, _, _ = self.lang_sam_model.predict(image_pil, self.text_prompt)
            # convert mask to numpy
            masks_np = [mask.squeeze().cpu().numpy() for mask in masks]

            if masks_np[0] is not None:
                metric_estimation = metric_estimation[~masks_np[0]] 
            else:
                print(f"No mask found for image {idx}!")

        # Create the point cloud
        x, y = np.meshgrid(np.arange(self.target_image_size[1]), np.arange(self.target_image_size[0]))
        # Normalize coordinates using the intrinsic matrix parameters
        focal_length_x = self.K[0, 0]
        focal_length_y = self.K[1, 1]
        c_x = self.K[0, 2]
        c_y = self.K[1, 2]
        x_normalized = (x - c_x) / focal_length_x
        y_normalized = (y - c_y) / focal_length_y
        z = np.array(metric_estimation)
        colors = left_image.reshape(-1, 3)
        # Calculate 3D points by unprojecting
        points = np.stack((np.multiply(x_normalized, z), np.multiply(y_normalized, z), z), axis=-1).reshape(-1, 3)
        # Convert points to homogeneous coordinates
        points = np.hstack((points, np.ones((points.shape[0], 1))))
        mask_valid = points[:, 2] < 45    # Discard all points further than 45 meters
        # Convert points from camera coordinates to velodyne coordinates
        points = self.camera_to_velodyne.dot(points.T).T
        # Get (x, y, z) coordinates
        points = points[:, :3]

        if self.use_sky_masks:
            mask_valid = mask_valid & (~get_mask(self.masks_path, self.sequence_num_id, idx, left_image.shape)).flatten()    # Invert mask
            mask_edges = edge_filter(torch.from_numpy(metric_estimation), threshold=0.2, neighborhood="8").detach().cpu().numpy()
            mask_valid = mask_valid & mask_edges.flatten()
        points = points[mask_valid, :]
        colors = colors[mask_valid, :]

        # save_point_cloud_to_obj(points, 'point_cloud_' + str(idx), colors)
        return points
    