import datetime

# from dotdict import dotdict
import glob
import os
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

# from dotdict import DotDict as dotdict
from dotdict import dotdict
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter


class AFMKITTI360Dataset(Dataset):
    def __init__(
        self,
        data_path: str,
        pose_path: str,
        sequence_id: str,
        split_path: Optional[str],
        mode: str = "lidar",  # LiDAR mode: 'lidar', 'lidar_mono_left', 'lidar_mono_right, 'lidar_stereo'
        target_image_size=(376, 1408),  # Size of rectified images
        return_stereo=False,
        return_depth=False,
        return_scans=False,  # True to use LiDAR data
        frame_count=2,
        keyframe_offset=0,
        dilation=1,
        eigen_depth=True,
        color_aug=False,
        is_preprocessed=False,
    ):
        self.data_path = data_path
        self.data_dir = data_path
        self.pose_path = pose_path
        self.split_path = split_path
        self.mode = mode
        self.target_image_size = target_image_size
        self.return_stereo = return_stereo
        self.return_depth = return_depth
        self.return_scans = return_scans
        self.frame_count = frame_count
        self.dilation = dilation
        self.keyframe_offset = keyframe_offset
        self.eigen_depth = eigen_depth
        self.color_aug = color_aug
        self.is_preprocessed = is_preprocessed

        # Filter sequences that have the requested id in their string
        all_sequences = self._get_sequences(self.data_path)
        self._sequence = [seq for seq in all_sequences if sequence_id in seq]
        if len(self._sequence) == 0:
            raise ValueError(f"Sequence {sequence_id} not found in dataset.")
        self.sequence_id = self._sequence[0]
        self.sequence_num_id = sequence_id
        self._calibs = self._load_calibs(self.data_path)

        # LiDAR data
        velodyne_dir = os.path.join(
            self.data_path,
            "data_3d_raw",
            self._sequence[0],
            "velodyne_points",
        )
        # Extract the datapoints ids from the scan_files
        self.scan_files = sorted(glob.glob(velodyne_dir + "/data/*.bin"))
        self._datapoints_ids = [
            int(os.path.basename(file).split(".")[0]) for file in self.scan_files
        ]
        # self._poses are the ground truth poses
        self._img_ids, self._poses = self._load_poses(self.pose_path, self._sequence)
        self.gt_poses = (
            self.load_poses()
        )  # Ground truth poses prepared for KISS-ICP pipeline

        self._left_offset = (
            (self.frame_count - 1) // 2 + self.keyframe_offset
        ) * self.dilation

        # Rectified pinhole cameras
        self._perspective_folder = (
            "data_rect"
            if not self.is_preprocessed
            else f"data_{self.target_image_size[0]}x{self.target_image_size[1]}"
        )

        # datapoints is a list with (sequence_id, frame_ids, l/r)
        self._datapoints = self.scan_files
        self._skip = 0
        self.length = len(self._datapoints)

        # Add correction for KITTI datasets, can be easilty removed if unwanted
        #  NOT REALLY SURE IF WE HAVE TO USE THIS. KISSICP DOES THIS BUT FOR THE KITTI ODOMETRY DATA
        from kiss_icp.pybind import kiss_icp_pybind

        self.correct_kitti_scan = lambda frame: np.asarray(
            kiss_icp_pybind._correct_kitti_scan(kiss_icp_pybind._Vector3dVector(frame))
        )

    # Read timestamps
    def get_frames_timestamps(self) -> np.ndarray:
        if self.return_scans:
            timestamps_dir = os.path.join(
                self.data_path, "data_3d_raw", self._sequence[0], "velodyne_points"
            )
            # Read every line of the timestamps file in a string
            with open(os.path.join(timestamps_dir, "timestamps.txt"), "r") as file:
                timestamps = file.readlines()
            # Eliminate the '\n' character at the end of each line
            timestamps = [ts[:-1] for ts in timestamps]
            # Split the timestamps in two parts: before and after the decimal dot for the seconds
            timestamps_before_dot, timestamps_after_dot = zip(
                *[ts_str.split(".", 1) for ts_str in timestamps]
            )
            timestamps_seconds = np.array(
                [
                    datetime.datetime.strptime(
                        timestamp, "%Y-%m-%d %H:%M:%S"
                    ).timestamp()
                    for timestamp in timestamps_before_dot
                ]
            )
            decimal_part = np.array(["0." + ts for ts in timestamps_after_dot]).astype(
                np.float64
            )
            # Compute final timestamps
            timestamps = timestamps_seconds + decimal_part
            # Calculate the elapsed time in seconds from the first measurement
            elapsed_time = timestamps - timestamps[0]
        else:
            # Note: we use only the left camera timestamps
            timestamps_dir = os.path.join(
                self.data_path, "data_2d_raw", self._sequence[0], "image_00"
            )
            # Read every line of the timestamps file in a string
            with open(os.path.join(timestamps_dir, "timestamps.txt"), "r") as file:
                timestamps = file.readlines()
            # Eliminate the '\n' character at the end of each line
            timestamps = [ts[:-1] for ts in timestamps]
            # Split the timestamps in two parts: before and after the decimal dot for the seconds
            timestamps_before_dot, timestamps_after_dot = zip(
                *[ts_str.split(".", 1) for ts_str in timestamps]
            )
            timestamps_seconds = np.array(
                [
                    datetime.datetime.strptime(
                        timestamp, "%Y-%m-%d %H:%M:%S"
                    ).timestamp()
                    for timestamp in timestamps_before_dot
                ]
            )
            decimal_part = np.array(["0." + ts for ts in timestamps_after_dot]).astype(
                np.float64
            )
            # Compute final timestamps
            timestamps = timestamps_seconds + decimal_part
            # Calculate the elapsed time in seconds from the first measurement
            elapsed_time = timestamps - timestamps[0]
        return elapsed_time

    # Read raw LiDAR scan data
    def read_point_cloud(self, scan_file: str):
        points = (
            np.fromfile(scan_file, dtype=np.float32)
            .reshape((-1, 4))[:, :3]
            .astype(np.float64)
        )
        # points = points[points[:, 2] > -2.9]        # Remove the annoying reflections. Note: this was already a comment in KISS-ICP code
        points = self.correct_kitti_scan(
            points
        )  # This is how it is done in KISS-ICP loader
        if self.mode == "lidar":
            # Return full lidar data
            return points
        else:
            # Return data that is in the camera(s) field of view
            points = points[self._get_valid_indexes(points, self.mode), :]
            return points

    def scans(self, idx):
        return self.read_point_cloud(self.scan_files[idx])

    def _get_valid_indexes(self, points: np.ndarray, mode: str) -> np.ndarray:
        """
        Filter points that are outside of the cameras image planes

        Args:
            points (np.ndarray): Points in the LiDAR frame

        Returns:
            np.ndarray: Filtered points
        """
        # If points second dimensions has 3 elements, add a fourth element with value 1
        if points.shape[1] == 3:
            points = np.hstack(
                (points, np.ones((points.shape[0], 1), dtype=np.float64))
            )
        points[:, 3] = 1.0
        K = self._calibs["K_perspective"]
        # Choose correct camera matrix and project the points to the cameras
        if mode == "lidar_mono_left" or mode == "lidar_mono_right":
            val_inds = None
            if mode == "lidar_mono_left":
                T_velo_to_cam = self._calibs["T_velo_to_cam"]["00"]
                velo_pts_im = np.dot(K @ T_velo_to_cam[:3, :], points.T).T
            elif mode == "lidar_mono_right":
                T_velo_to_cam = self._calibs["T_velo_to_cam"]["01"]
                velo_pts_im = np.dot(K @ T_velo_to_cam[:3, :], points.T).T
            # Project the points to the camera
            velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., None]

            # The projection is normalized to [-1, 1] -> transform to [0, height-1] x [0, width-1]
            velo_pts_im[:, 0] = np.round(
                (velo_pts_im[:, 0] * 0.5 + 0.5) * self.target_image_size[1]
            )
            velo_pts_im[:, 1] = np.round(
                (velo_pts_im[:, 1] * 0.5 + 0.5) * self.target_image_size[0]
            )

            # Check if in bounds
            val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
            val_inds = (
                val_inds
                & (velo_pts_im[:, 0] < self.target_image_size[1])
                & (velo_pts_im[:, 1] < self.target_image_size[0])
            )
            # Get points that lie in front of the image plane
            val_inds = val_inds & (velo_pts_im[:, 2] > 0)
        elif mode == "lidar_stereo":
            valid_left_indexes = self._get_valid_indexes(points, mode="lidar_mono_left")
            valid_right_indexes = self._get_valid_indexes(
                points, mode="lidar_mono_right"
            )
            val_inds = valid_left_indexes | valid_right_indexes
        else:
            raise ValueError(f"Invalid mode: {mode}")
        return val_inds

    def get_points(self, pose: torch.Tensor) -> torch.Tensor:
        """Get points from a pose.

        Args:
            pose (torch.Tensor): Pose of shape (4, 4)

        Returns:
            torch.Tensor: Points of shape (N, 3). NOTE: the points are in the world coordinate system.
        """
        OUT_RES = dotdict(
            X_RANGE=(-9, 9),
            Y_RANGE=(0.0, 0.75),
            Z_RANGE=(21, 3),
            X_RES=256,
            Y_RES=64,
            Z_RES=256,
        )

        cam_incl_adjust = torch.tensor(
            [
                [1.0000000, 0.0000000, 0.0000000, 0],
                [0.0000000, 0.9961947, -0.0871557, 0],
                [0.0000000, 0.0871557, 0.9961947, 0],
                [0.0000000, 000000000, 0.0000000, 1],
            ],
            dtype=torch.float32,
        ).view(4, 4)

        points = regular_grid(
            OUT_RES.X_RANGE,
            OUT_RES.Y_RANGE,
            OUT_RES.Z_RANGE,
            OUT_RES.X_RES,
            OUT_RES.Y_RES,
            OUT_RES.Z_RES,
            cam_incl_adjust=cam_incl_adjust,
        )
        return points

    def check_file_integrity(self, seq, id):
        dp = Path(self.data_path)
        image_00 = dp / "data_2d_raw" / seq / "image_00" / self._perspective_folder
        image_01 = dp / "data_2d_raw" / seq / "image_01" / self._perspective_folder

        seq_len = self._img_ids[seq].shape[0]

        ids = [id] + [
            max(min(i, seq_len - 1), 0)
            for i in range(
                id - self._left_offset,
                id - self._left_offset + self.frame_count * self.dilation,
                self.dilation,
            )
            if i != id
        ]

        img_ids = [self.get_img_id_from_id(seq, id) for id in ids]

        for img_id in img_ids:
            if not (
                (image_00 / f"{img_id:010d}.png").exists()
                and (image_01 / f"{img_id:010d}.png").exists()
            ):
                return False
        return True

    @staticmethod
    def _get_sequences(data_path):
        all_sequences = []

        seqs_path = Path(data_path) / "data_2d_raw"
        for seq in seqs_path.iterdir():
            if not seq.is_dir():
                continue
            all_sequences.append(seq.name)

        return all_sequences

    @staticmethod
    def _load_calibs(data_path):
        data_path = Path(data_path)

        calib_folder = data_path / "calibration"
        cam_to_pose_file = calib_folder / "calib_cam_to_pose.txt"
        cam_to_velo_file = calib_folder / "calib_cam_to_velo.txt"
        intrinsics_file = calib_folder / "perspective.txt"

        cam_to_pose_data = {}
        with open(cam_to_pose_file, "r") as f:
            for line in f.readlines():
                key, value = line.split(":", 1)
                try:
                    cam_to_pose_data[key] = np.array(
                        [float(x) for x in value.split()], dtype=np.float32
                    )
                except ValueError:
                    pass

        cam_to_velo_data = None
        with open(cam_to_velo_file, "r") as f:
            line = f.readline()
            try:
                cam_to_velo_data = np.array(
                    [float(x) for x in line.split()], dtype=np.float32
                )
            except ValueError:
                pass

        intrinsics_data = {}
        with open(intrinsics_file, "r") as f:
            for line in f.readlines():
                key, value = line.split(":", 1)
                try:
                    intrinsics_data[key] = np.array(
                        [float(x) for x in value.split()], dtype=np.float32
                    )
                except ValueError:
                    pass

        im_size_rect = (
            int(intrinsics_data["S_rect_00"][1]),
            int(intrinsics_data["S_rect_00"][0]),
        )

        # Projection matrices
        # We use these projection matrices.
        # This makes downstream processing easier, but it could be done differently.
        P_rect_00 = np.reshape(intrinsics_data["P_rect_00"], (3, 4))
        P_rect_01 = np.reshape(intrinsics_data["P_rect_01"], (3, 4))

        # Rotation matrices from raw to rectified -> Needs to be inverted later
        R_rect_00 = np.eye(4, dtype=np.float32)
        R_rect_01 = np.eye(4, dtype=np.float32)
        R_rect_00[:3, :3] = np.reshape(intrinsics_data["R_rect_00"], (3, 3))
        R_rect_01[:3, :3] = np.reshape(intrinsics_data["R_rect_01"], (3, 3))

        # Load cam to pose transforms
        T_00_to_pose = np.eye(4, dtype=np.float32)
        T_01_to_pose = np.eye(4, dtype=np.float32)
        T_02_to_pose = np.eye(4, dtype=np.float32)
        T_03_to_pose = np.eye(4, dtype=np.float32)
        T_00_to_velo = np.eye(4, dtype=np.float32)

        T_00_to_pose[:3, :] = np.reshape(cam_to_pose_data["image_00"], (3, 4))
        T_01_to_pose[:3, :] = np.reshape(cam_to_pose_data["image_01"], (3, 4))
        T_02_to_pose[:3, :] = np.reshape(cam_to_pose_data["image_02"], (3, 4))
        T_03_to_pose[:3, :] = np.reshape(cam_to_pose_data["image_03"], (3, 4))
        T_00_to_velo[:3, :] = np.reshape(cam_to_velo_data, (3, 4))

        # Compute cam to pose transforms for rectified perspective cameras
        T_rect_00_to_pose = T_00_to_pose @ np.linalg.inv(R_rect_00)
        T_rect_01_to_pose = T_01_to_pose @ np.linalg.inv(R_rect_01)

        # Compute velo to cameras and velo to pose transforms
        T_velo_to_rect_00 = R_rect_00 @ np.linalg.inv(T_00_to_velo)
        T_velo_to_pose = T_rect_00_to_pose @ T_velo_to_rect_00
        T_pose_to_velo = np.linalg.inv(T_velo_to_pose)
        T_velo_to_rect_01 = np.linalg.inv(T_rect_01_to_pose) @ T_velo_to_pose

        # Calibration matrix is the same for both perspective cameras
        K = P_rect_00[:3, :3]

        # Normalize calibration (so that calibration matrix is independet from image resolution)
        f_x = K[0, 0] / im_size_rect[1]
        f_y = K[1, 1] / im_size_rect[0]
        c_x = K[0, 2] / im_size_rect[1]
        c_y = K[1, 2] / im_size_rect[0]

        # Change to image coordinates [-1, 1]
        K[0, 0] = f_x * 2.0
        K[1, 1] = f_y * 2.0
        K[0, 2] = c_x * 2.0 - 1
        K[1, 2] = c_y * 2.0 - 1

        # Use same camera calibration as perspective cameras for resampling

        # Read calibration time
        calib_time = ""
        with open(cam_to_pose_file, "r") as f:
            for line in f.readlines():
                key, value = line.split(":", 1)
                if key == "calib_time":
                    calib_time = value

        calibs = {
            "K_perspective": K,
            "T_cam_to_pose": {
                "00": T_rect_00_to_pose,
                "01": T_rect_01_to_pose,
                "02": T_02_to_pose,
                "03": T_03_to_pose,
            },
            "T_velo_to_cam": {
                "00": T_velo_to_rect_00,
                "01": T_velo_to_rect_01,
            },
            "T_velo_to_pose": T_velo_to_pose,
            "T_pose_to_velo": T_pose_to_velo,
            "im_size": im_size_rect,
            "calib_time": calib_time,
        }

        return calibs

    # def apply_calibration(self, poses: np.ndarray) -> np.ndarray:
    #     """Converts from Velodyne to Pose Frame"""
    #     Tr =  self._calibs["T_pose_to_velo"]
    #     return Tr @ poses @ np.linalg.inv(Tr)

    @staticmethod
    def _load_poses(pose_path, sequences):
        ids = {}
        poses = {}

        for seq in sequences:
            pose_file = Path(pose_path) / seq / f"poses.txt"

            try:
                pose_data = np.loadtxt(pose_file)
            except FileNotFoundError:
                print(f"Ground truth poses are not avaialble for sequence {seq}.")

            ids_seq = pose_data[:, 0].astype(int)
            poses_seq = pose_data[:, 1:].astype(np.float32).reshape((-1, 3, 4))
            poses_seq = np.concatenate(
                (poses_seq, np.zeros_like(poses_seq[:, :1, :])), axis=1
            )
            poses_seq[:, 3, 3] = 1

            ids[seq] = ids_seq
            poses[seq] = poses_seq
        return ids, poses

    def load_poses(self):
        # Return ground truth poses in LiDAR coordinate system
        def _lidar_pose_gt(poses_gt):
            tr_inv = self._calibs["T_velo_to_pose"]
            tr = self._calibs["T_pose_to_velo"]  # 4x4
            # Get first gt pose that has a corresponding datapoint (some sequences have no datapoints for the first gt poses)
            datapoints_ids = np.array(
                self.get_datapoints_ids()
            )  # Get the ids that relate datapoints
            gt_poses_ids = np.array(
                self.get_gt_poses_ids()
            )  # Get the ids that relate GT poses to datapoints
            ids_to_compare = np.intersect1d(datapoints_ids, gt_poses_ids)
            gt_poses_indexes_to_compare = gt_poses_ids[
                np.in1d(gt_poses_ids, ids_to_compare, assume_unique=True)
            ]
            new_poses = np.einsum("...ij,jk->...ik", poses_gt, tr_inv)
            first_gt_pose_id_with_correspondance = np.where(
                np.in1d(gt_poses_ids, gt_poses_indexes_to_compare, assume_unique=True)
            )[0][0]
            # Discard all gt poses that are before the first datapoint
            new_poses = new_poses[first_gt_pose_id_with_correspondance:]
            self._img_ids[self._sequence[0]] = self._img_ids[self._sequence[0]][
                first_gt_pose_id_with_correspondance:
            ]
            return np.einsum("ij,...jk->...ik", np.linalg.inv(new_poses[0]), new_poses)

        poses = self._poses[self._sequence[0]]
        return _lidar_pose_gt(poses)

    def get_gt_poses_ids(self):
        return self._img_ids[self._sequence[0]]

    def get_datapoints_ids(self):
        return self._datapoints_ids

    def get_img_id_from_id(self, sequence, id):
        return self._img_ids[sequence][id]

    def load_images(self, seq, img_ids, load_left, load_right):
        imgs_p_left = []
        imgs_p_right = []

        for id in img_ids:
            if load_left:
                img_perspective = (
                    cv2.cvtColor(
                        cv2.imread(
                            os.path.join(
                                self.data_path,
                                "data_2d_raw",
                                seq,
                                "image_00",
                                self._perspective_folder,
                                f"{id:010d}.png",
                            )
                        ),
                        cv2.COLOR_BGR2RGB,
                    ).astype(np.float32)
                    / 255
                )
                imgs_p_left += [img_perspective]

            if load_right:
                img_perspective = (
                    cv2.cvtColor(
                        cv2.imread(
                            os.path.join(
                                self.data_path,
                                "data_2d_raw",
                                seq,
                                "image_01",
                                self._perspective_folder,
                                f"{id:010d}.png",
                            )
                        ),
                        cv2.COLOR_BGR2RGB,
                    ).astype(np.float32)
                    / 255
                )
                imgs_p_right += [img_perspective]

        return imgs_p_left, imgs_p_right

    def process_img(
        self,
        img: np.array,
        color_aug_fn=None,
    ):
        if self.target_image_size:
            img = cv2.resize(
                img,
                (self.target_image_size[1], self.target_image_size[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        img = np.transpose(img, (2, 0, 1))
        img = torch.tensor(img)

        if color_aug_fn is not None:
            img = color_aug_fn(img)

        img = img * 2 - 1
        return img

    def load_depth(self, seq, img_id, is_right):
        points = np.fromfile(
            os.path.join(
                self.data_path,
                "data_3d_raw",
                seq,
                "velodyne_points",
                "data",
                f"{img_id:010d}.bin",
            ),
            dtype=np.float32,
        ).reshape(-1, 4)
        points[:, 3] = 1.0

        T_velo_to_cam = self._calibs["T_velo_to_cam"]["00" if not is_right else "01"]
        K = self._calibs["K_perspective"]

        # project the points to the camera
        velo_pts_im = np.dot(K @ T_velo_to_cam[:3, :], points.T).T
        velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., None]

        # the projection is normalized to [-1, 1] -> transform to [0, height-1] x [0, width-1]
        velo_pts_im[:, 0] = np.round(
            (velo_pts_im[:, 0] * 0.5 + 0.5) * self.target_image_size[1]
        )
        velo_pts_im[:, 1] = np.round(
            (velo_pts_im[:, 1] * 0.5 + 0.5) * self.target_image_size[0]
        )

        # check if in bounds
        val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
        val_inds = (
            val_inds
            & (velo_pts_im[:, 0] < self.target_image_size[1])
            & (velo_pts_im[:, 1] < self.target_image_size[0])
        )
        velo_pts_im = velo_pts_im[val_inds, :]

        # project to image
        depth = np.zeros(self.target_image_size)
        depth[
            velo_pts_im[:, 1].astype(np.int32), velo_pts_im[:, 0].astype(np.int32)
        ] = velo_pts_im[:, 2]

        # find the duplicate points and choose the closest depth
        inds = (
            velo_pts_im[:, 1] * (self.target_image_size[1] - 1) + velo_pts_im[:, 0] - 1
        )
        dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
        for dd in dupe_inds:
            pts = np.where(inds == dd)[0]
            x_loc = int(velo_pts_im[pts[0], 0])
            y_loc = int(velo_pts_im[pts[0], 1])
            depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
        depth[depth < 0] = 0

        return depth[None, :, :]

    def __getitem__(self, index: int):
        if self.return_scans:
            return self.scans(index)
        data = {}  # dictionary to return
        _start_time = time.time()

        if index >= self.length:
            raise IndexError()

        if self._skip != 0:
            index += self._skip

        # Obtain data for the current frame
        sequence, id, is_right = self.sequence_id, index, self.return_stereo

        load_left, load_right = (
            not is_right
        ) or self.return_stereo, is_right or self.return_stereo

        ids = [id]
        img_ids = ids
        #[self.get_img_id_from_id(sequence, id) for id in ids]

        if False:
            color_aug_fn = get_color_aug_fn(
                ColorJitter.get_params(
                    brightness=(0.8, 1.2),
                    contrast=(0.8, 1.2),
                    saturation=(0.8, 1.2),
                    hue=(-0.1, 0.1),
                )
            )
        else:
            color_aug_fn = None

        _start_time_loading = time.time()
        imgs_p_left, imgs_p_right = self.load_images(
            sequence,
            img_ids,
            load_left,
            load_right,
        )
        _loading_time = np.array(time.time() - _start_time_loading)

        _start_time_processing = time.time()
        # Use no color augmentation
        # imgs_p_left = [
        #     self.process_img(img, color_aug_fn=color_aug_fn) for img in imgs_p_left
        # ]
        # imgs_p_right = [
        #     self.process_img(img, color_aug_fn=color_aug_fn) for img in imgs_p_right
        # ]
        _processing_time = np.array(time.time() - _start_time_processing)

        # These poses are camera to world !!
        # poses_p_left = (
        #     [
        #         self._poses[sequence][i, :, :] @ self._calibs["T_cam_to_pose"]["00"]
        #         for i in ids
        #     ]
        #     if load_left
        #     else []
        # )
        # poses_p_right = (
        #     [
        #         self._poses[sequence][i, :, :] @ self._calibs["T_cam_to_pose"]["01"]
        #         for i in ids
        #     ]
        #     if load_right
        #     else []
        # )

        projs_p_left = [self._calibs["K_perspective"] for _ in ids] if load_left else []
        projs_p_right = (
            [self._calibs["K_perspective"] for _ in ids] if load_right else []
        )

        imgs = (
            imgs_p_left + imgs_p_right if not is_right else imgs_p_right + imgs_p_left
        )
        projs = (
            projs_p_left + projs_p_right
            if not is_right
            else projs_p_right + projs_p_left
        )
        # Add poses
        # poses = (
        #     poses_p_left + poses_p_right
        #     if not is_right
        #     else poses_p_right + poses_p_left
        # )

        ids = np.array(ids + ids, dtype=np.int32)

        if self.return_depth:
            depths = [self.load_depth(sequence, index, is_right)]
        else:
            depths = []

        _proc_time = np.array(time.time() - _start_time)

        scans = self.scans(index)
        # print(_loading_time, _processing_time, _proc_time)

        data = {
            "imgs": imgs,  # left and right images in a list ([0] for left, [1] for right)
            "projs": projs,  # left and right projection matrices in a list ([0] for left, [1] for right)
            # "poses": poses,  # left and right GT poses (camera to world) in a list ([0] for left, [1] for right)
            "depths": depths,  # depth maps projected on image using LiDAR data (not done yet)
            "scans": scans,  # 3D LiDAR data in a list
            "ts": ids,
            "t__get_item__": np.array([_proc_time]),
            "index": np.array([index]),
        }

        return data

    def __len__(self) -> int:
        return self.length
