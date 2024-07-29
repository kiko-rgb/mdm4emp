import logging
import os
import time
from pathlib import Path
from dotdict import dotdict
import yaml

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter

from bts.common.point_sampling import regular_grid
from bts.common.sampling_strategies import get_encoder_sampling
from bts.common.augmentation import get_color_aug_fn
from .common import lidar_to_depth_map, load_image, load_lidar_points

logger = logging.getLogger("dataset")


class FisheyeToPinholeSampler:
    def __init__(self, K_target, target_image_size, calibs, rotation=None):
        self._compute_transform(K_target, target_image_size, calibs, rotation)

    def _compute_transform(self, K_target, target_image_size, calibs, rotation=None):
        x = (
            torch.linspace(-1, 1, target_image_size[1])
            .view(1, -1)
            .expand(target_image_size)
        )
        y = (
            torch.linspace(-1, 1, target_image_size[0])
            .view(-1, 1)
            .expand(target_image_size)
        )
        z = torch.ones_like(x)
        xyz = torch.stack((x, y, z), dim=-1).view(-1, 3)

        # Unproject
        xyz = (torch.inverse(torch.tensor(K_target)) @ xyz.T).T

        if rotation is not None:
            xyz = (torch.tensor(rotation) @ xyz.T).T

        # Backproject into fisheye
        xyz = xyz / torch.norm(xyz, dim=-1, keepdim=True)
        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]

        xi_src = calibs["mirror_parameters"]["xi"]
        x = x / (z + xi_src)
        y = y / (z + xi_src)

        k1 = calibs["distortion_parameters"]["k1"]
        k2 = calibs["distortion_parameters"]["k2"]

        r = x * x + y * y
        factor = 1 + k1 * r + k2 * r * r
        x = x * factor
        y = y * factor

        gamma0 = calibs["projection_parameters"]["gamma1"]
        gamma1 = calibs["projection_parameters"]["gamma2"]
        u0 = calibs["projection_parameters"]["u0"]
        v0 = calibs["projection_parameters"]["v0"]

        x = x * gamma0 + u0
        y = y * gamma1 + v0

        xy = torch.stack((x, y), dim=-1).view(1, *target_image_size, 2)
        self.sample_pts = xy

    def resample(self, img):
        img = img.unsqueeze(0)
        resampled_img = F.grid_sample(img, self.sample_pts, align_corners=True).squeeze(
            0
        )
        return resampled_img


class OldKITTI360Dataset(Dataset):
    def __init__(
        self,
        data_path: Path,
        pose_path: Path,
        segmentation_path: Path | None,
        split_path: Path | None,
        target_image_size=(192, 640),
        return_stereo=False,
        return_depth=False,
        return_fisheye=True,  ## default: True
        return_3d_bboxes=False,
        return_segmentation=False,
        frame_count=2,
        fisheye_rotation=0,
        fisheye_offset: list[int] = [1],
        pinhole_offset: list[int] = [1],
        eigen_depth=True,
        color_aug=False,
        is_preprocessed=False,
    ):
        self.data_path = data_path
        self.pose_path = pose_path
        self.segmentation_path = segmentation_path
        self.split_path = split_path
        self.target_image_size = target_image_size
        self.return_stereo = return_stereo
        self.return_fisheye = return_fisheye
        self.return_depth = return_depth
        self.return_3d_bboxes = return_3d_bboxes
        self.return_segmentation = (
            return_segmentation and self.segmentation_path is not None
        )
        self.frame_count = frame_count
        self.fisheye_rotation = fisheye_rotation
        self.fisheye_offset = fisheye_offset
        self.stereo_offset = pinhole_offset
        self.eigen_depth = eigen_depth
        self.color_aug = color_aug
        self.is_preprocessed = is_preprocessed

        if isinstance(self.fisheye_rotation, float) or isinstance(
            self.fisheye_rotation, int
        ):
            self.fisheye_rotation = (0, self.fisheye_rotation)
        self.fisheye_rotation = tuple(self.fisheye_rotation)

        self.fisheye_offset = fisheye_offset
        self.stereo_offset = pinhole_offset

        self._sequences = self._get_sequences(self.data_path)

        self._calibs = self._load_calibs(self.data_path, self.fisheye_rotation)

        self._img_ids, self._poses = self._load_poses(self.pose_path, self._sequences)
        self._left_offset = (self.frame_count - 1) // 2

        self._perspective_folder = (
            "data_rect"
            if not self.is_preprocessed
            else f"data_{self.target_image_size[0]}x{self.target_image_size[1]}"
        )
        self._fisheye_folder = (
            "data_rgb"
            if not self.is_preprocessed
            else f"data_{self.target_image_size[0]}x{self.target_image_size[1]}_{self.fisheye_rotation[0]}x{self.fisheye_rotation[1]}"
        )
        self._segmentation_perspective_folder = "data_192x640"
        self._segmentation_fisheye_folder = "data_192x640_0x-15"

        if self.split_path is not None:
            self._datapoints = self._load_split(self.split_path, self._img_ids)
        elif self.return_segmentation:
            self._datapoints = self._semantics_split(
                self._sequences, self.data_path, self._img_ids
            )
        else:
            self._datapoints = self._full_split(
                self._sequences, self._img_ids, self.check_file_integrity
            )

        if self.return_segmentation:
            # Segmentations are only provided for the left camera
            self._datapoints = [dp for dp in self._datapoints if not dp[2]]

        self._skip = 0
        self.length = len(self._datapoints)

        # TODO: add to config
        self.encoder_sampling = get_encoder_sampling(
            {
                "name": "kitti_360_stereo",
                "args": {
                    "num_encoder_frames": 1,
                    "num_stereo_frames": 2,
                    "always_use_base_frame": True,
                },
            }
        )

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
        dp = self.data_path
        image_00 = dp / "data_2d_raw" / seq / "image_00" / self._perspective_folder
        image_01 = dp / "data_2d_raw" / seq / "image_01" / self._perspective_folder
        image_02 = dp / "data_2d_raw" / seq / "image_02" / self._fisheye_folder
        image_03 = dp / "data_2d_raw" / seq / "image_03" / self._fisheye_folder

        seq_len = self._img_ids[seq].shape[0]

        ids = [id] + [
            max(min(i, seq_len - 1), 0)
            for i in range(
                id - self._left_offset,
                id - self._left_offset + self.frame_count,
            )
            if i != id
        ]
        ids_fish = [max(min(id + self.fisheye_offset, seq_len - 1), 0)] + [
            max(min(i, seq_len - 1), 0)
            for i in range(
                id + self.fisheye_offset - self._left_offset,
                id + self.fisheye_offset - self._left_offset + self.frame_count,
            )
            if i != id + self.fisheye_offset
        ]

        img_ids = [self.get_img_id_from_id(seq, id) for id in ids]
        img_ids_fish = [self.get_img_id_from_id(seq, id) for id in ids_fish]

        for img_id in img_ids:
            if not (
                (image_00 / f"{img_id:010d}.png").exists()
                and (image_01 / f"{img_id:010d}.png").exists()
            ):
                return False
        if self.return_fisheye:
            for img_id in img_ids_fish:
                if not (
                    (image_02 / f"{img_id:010d}.png").exists()
                    and (image_03 / f"{img_id:010d}.png").exists()
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
    def _full_split(sequences, img_ids, check_integrity):
        datapoints = []
        for seq in sorted(sequences):
            ids = [id for id in range(len(img_ids[seq])) if check_integrity(seq, id)]
            datapoints_seq = [(seq, id, False) for id in ids] + [
                (seq, id, True) for id in ids
            ]
            datapoints.extend(datapoints_seq)
        return datapoints

    @staticmethod
    def _semantics_split(sequences, data_path, img_ids):
        datapoints = []
        for seq in sorted(sequences):
            datapoints_seq = [(seq, id, False) for id in range(len(img_ids[seq]))]
            datapoints_seq = [
                dp
                for dp in datapoints_seq
                if os.path.exists(
                    os.path.join(
                        data_path,
                        "data_2d_semantics",
                        "train",
                        seq,
                        "image_00",
                        "semantic_rgb",
                        f"{img_ids[seq][dp[1]]:010d}.png",
                    )
                )
            ]
            datapoints.extend(datapoints_seq)
        return datapoints

    @staticmethod
    def _load_split(split_path, img_ids):
        img_id2id = {
            seq: {id: i for i, id in enumerate(ids)} for seq, ids in img_ids.items()
        }

        with open(split_path, "r") as f:
            lines = f.readlines()

        def split_line(l):
            segments = l.split(" ")
            seq = segments[0]
            id = img_id2id[seq][int(segments[1])]
            return seq, id, segments[2][0] == "r"

        return list(map(split_line, lines))

    @staticmethod
    def _load_calibs(data_path: Path, fisheye_rotation: tuple = (0.0, 0.0)):
        calib_folder = data_path / "calibration"
        cam_to_pose_file = calib_folder / "calib_cam_to_pose.txt"
        cam_to_velo_file = calib_folder / "calib_cam_to_velo.txt"
        intrinsics_file = calib_folder / "perspective.txt"
        fisheye_02_file = calib_folder / "image_02.yaml"
        fisheye_03_file = calib_folder / "image_03.yaml"

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

        with open(fisheye_02_file, "r") as f:
            f.readline()  # Skips first line that defines the YAML version
            fisheye_02_data = yaml.safe_load(f)

        with open(fisheye_03_file, "r") as f:
            f.readline()  # Skips first line that defines the YAML version
            fisheye_03_data = yaml.safe_load(f)

        im_size_rect = (
            int(intrinsics_data["S_rect_00"][1]),
            int(intrinsics_data["S_rect_00"][0]),
        )
        im_size_fish = (fisheye_02_data["image_height"], fisheye_02_data["image_width"])

        # Projection matrices
        # We use these projection matrices also when resampling the fisheye cameras.
        # This makes downstream processing easier, but it could be done differently.
        proj_rect_00 = np.reshape(intrinsics_data["P_rect_00"], (3, 4))
        proj_rect_01 = np.reshape(intrinsics_data["P_rect_01"], (3, 4))

        # Rotation matrices from raw to rectified -> Needs to be inverted later
        rotation_rect_00 = np.eye(4, dtype=np.float32)
        rotation_rect_01 = np.eye(4, dtype=np.float32)
        rotation_rect_00[:3, :3] = np.reshape(intrinsics_data["R_rect_00"], (3, 3))
        rotation_rect_01[:3, :3] = np.reshape(intrinsics_data["R_rect_01"], (3, 3))

        # Rotation matrices from resampled fisheye to raw fisheye
        # TODO: this is dummy
        fisheye_rotation = [0, 0]
        fisheye_rotation = np.array(fisheye_rotation).reshape((1, 2))
        R_02 = np.eye(4, dtype=np.float32)
        R_03 = np.eye(4, dtype=np.float32)
        R_02[:3, :3] = (
            Rotation.from_euler("xy", fisheye_rotation[:, [1, 0]], degrees=True)
            .as_matrix()
            .astype(np.float32)
        )
        R_03[:3, :3] = (
            Rotation.from_euler(
                "xy", fisheye_rotation[:, [1, 0]] * np.array([[1, -1]]), degrees=True
            )
            .as_matrix()
            .astype(np.float32)
        )

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
        T_rect_00_to_pose = T_00_to_pose @ np.linalg.inv(rotation_rect_00)
        T_rect_01_to_pose = T_01_to_pose @ np.linalg.inv(rotation_rect_01)

        # Compute cam to pose transform for fisheye cameras
        T_02_to_pose = T_02_to_pose @ R_02
        T_03_to_pose = T_03_to_pose @ R_03

        # Compute velo to cameras and velo to pose transforms
        T_velo_to_rect_00 = rotation_rect_00 @ np.linalg.inv(T_00_to_velo)
        T_velo_to_pose = T_rect_00_to_pose @ T_velo_to_rect_00
        T_velo_to_rect_01 = np.linalg.inv(T_rect_01_to_pose) @ T_velo_to_pose

        calibs = {
            "K_00": proj_rect_00[:3, :3],
            "K_01": proj_rect_01[:3, :3],
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
            "fisheye": {
                "calib_02": fisheye_02_data,
                "calib_03": fisheye_03_data,
                "R_02": R_02[:3, :3],
                "R_03": R_03[:3, :3],
            },
            "im_size": im_size_rect,
        }

        return calibs

    @staticmethod
    def _get_resamplers(calibs, target_image_size):
        resampler_02 = FisheyeToPinholeSampler(
            calibs["K_00"],
            target_image_size,
            calibs["fisheye"]["calib_02"],
            calibs["fisheye"]["R_02"],
        )
        resampler_03 = FisheyeToPinholeSampler(
            calibs["K_01"],
            target_image_size,
            calibs["fisheye"]["calib_03"],
            calibs["fisheye"]["R_03"],
        )

        return resampler_02, resampler_03

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

    def get_img_id_from_id(self, sequence, id):
        return self._img_ids[sequence][id]

    def load_images(
        self, seq, img_ids, load_left, load_right, img_ids_fish=None
    ) -> tuple[
        list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]
    ]:
        imgs_p_left: list[torch.Tensor] = []
        imgs_f_left: list[torch.Tensor] = []
        imgs_p_right: list[torch.Tensor] = []
        imgs_f_right: list[torch.Tensor] = []

        if img_ids_fish is None:
            img_ids_fish = img_ids

        for id in img_ids:
            if load_left:
                img_perspective = load_image(
                    self.data_path.joinpath(
                        "data_2d_raw",
                        seq,
                        "image_00",
                        self._perspective_folder,
                        f"{id:010d}.png",
                    )
                )
                imgs_p_left += [img_perspective]

            if load_right:
                img_perspective = load_image(
                    self.data_path.joinpath(
                        "data_2d_raw",
                        seq,
                        "image_01",
                        self._perspective_folder,
                        f"{id:010d}.png",
                    )
                )
                imgs_p_right += [img_perspective]

        for id in img_ids_fish:
            if load_left:
                img_fisheye = load_image(
                    self.data_path.joinpath(
                        "data_2d_raw",
                        seq,
                        "image_02",
                        self._fisheye_folder,
                        f"{id:010d}.png",
                    )
                )
                imgs_f_left += [img_fisheye]
            if load_right:
                img_fisheye = load_image(
                    self.data_path.joinpath(
                        "data_2d_raw",
                        seq,
                        "image_03",
                        self._fisheye_folder,
                        f"{id:010d}.png",
                    )
                )
                imgs_f_right += [img_fisheye]

        return imgs_p_left, imgs_f_left, imgs_p_right, imgs_f_right

    def load_segmentation_from_path(self, path: Path) -> torch.Tensor:
        seg = torch.Tensor(cv2.imread(path.as_posix(), cv2.IMREAD_UNCHANGED))
        return seg

    def load_segmentation_images(
        self, seq, img_ids, load_left, load_right, img_ids_fish=None
    ) -> tuple[
        list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]
    ]:
        imgs_p_left = []
        imgs_f_left = []
        imgs_p_right = []
        imgs_f_right = []

        if img_ids_fish is None:
            img_ids_fish = img_ids

        for id in img_ids:
            if load_left:
                img_perspective = self.load_segmentation_from_path(
                    self.segmentation_path.joinpath(
                        seq,
                        "image_00",
                        self._segmentation_perspective_folder,
                        f"{id:010d}.png",
                    )
                )
                imgs_p_left += [img_perspective]

            if load_right:

                img_perspective = self.load_segmentation_from_path(
                    self.segmentation_path.joinpath(
                        seq,
                        "image_01",
                        self._segmentation_perspective_folder,
                        f"{id:010d}.png",
                    )
                )
                imgs_p_right += [img_perspective]

        for id in img_ids_fish:
            if load_left:
                img_fisheye = self.load_segmentation_from_path(
                    self.data_segmentation_path.joinpath(
                        seq,
                        "image_02",
                        self._segmentation_fisheye_folder,
                        f"{id:010d}.png",
                    )
                )
                imgs_f_left += [img_fisheye]
            if load_right:
                img_fisheye = self.load_segmentation_from_path(
                    self.data_segmentation_path.joinpath(
                        seq,
                        "image_03",
                        self._segmentation_fisheye_folder,
                        f"{id:010d}.png",
                    )
                )
                imgs_f_right += [img_fisheye]

        return imgs_p_left, imgs_p_right, imgs_f_left, imgs_f_right

    def process_img(
        self,
        img: torch.Tensor,
        color_aug_fn=None,
        resampler: FisheyeToPinholeSampler | None = None,
    ):
        img = img.permute(2, 0, 1)
        if resampler is not None and not self.is_preprocessed:
            img = resampler.resample(img)

        if color_aug_fn is not None:
            img = color_aug_fn(img)

        return img

    # def load_segmentation(self, seq, img_id):
    #     seg = cv2.imread(
    #         os.path.join(
    #             self.data_path,
    #             "data_2d_semantics",
    #             "train",
    #             seq,
    #             "image_00",
    #             "semantic",
    #             f"{img_id:010d}.png",
    #         ),
    #         cv2.IMREAD_UNCHANGED,
    #     )
    #     seg = cv2.resize(
    #         seg,
    #         (self.target_image_size[1], self.target_image_size[0]),
    #         interpolation=cv2.INTER_NEAREST,
    #     )
    #     return seg

    def load_depth(self, seq, img_id, is_right, target_image_size):
        points = load_lidar_points(
            self.data_path.joinpath(
                "data_3d_raw", seq, "velodyne_points", "data", f"{img_id:010d}.bin"
            )
        )

        T_velo_to_cam = self._calibs["T_velo_to_cam"]["00" if not is_right else "01"]
        K = self._calibs["K_00" if not is_right else "K_01"]

        depth_map = lidar_to_depth_map(points, K, target_image_size, T_velo_to_cam)

        return depth_map[None, :, :]

    def __getitem__(self, index: int):
        _start_time = time.time()

        if index >= self.length:
            raise IndexError()

        if self._skip != 0:
            index += self._skip

        sequence, id, is_right = self._datapoints[index]
        seq_len = self._img_ids[sequence].shape[0]

        load_left, load_right = (
            not is_right
        ) or self.return_stereo, is_right or self.return_stereo

        def get_random_offsets(offsets: list[int], num_frames: int) -> list[int]:
            possible_ids = [
                id + offset
                for offset in offsets
                if 0 <= id + offset < seq_len - num_frames
            ]
            # Fall back to using the default offsets if no valid offsets are found
            if len(possible_ids) == 0:
                possible_ids = [idx for idx in range(id, min(seq_len, id + 20))]

            if len(possible_ids) > 0:
                rand_idx = torch.randint(0, len(possible_ids), (1,)).item()
                return [possible_ids[rand_idx] + offset for offset in range(num_frames)]
            else:
                logger.warning(
                    f"Could not find valid offsets for sequence {sequence} and id {id}. Using default offsets."
                )
                return [min(id + offset, seq_len) for offset in range(num_frames)]

        ## randomly sample fisheye in the time steps where it can see the occlusion with the stereo
        stereo_ids = [id] + get_random_offsets(self.stereo_offset, self.frame_count - 1)
        fisheye_ids = get_random_offsets(self.fisheye_offset, self.frame_count)

        ## and now ids_fish is 5 steps ahead of ids with 2 fisheye scenes
        img_ids = [self.get_img_id_from_id(sequence, id) for id in stereo_ids]
        img_ids_fish = [self.get_img_id_from_id(sequence, id) for id in fisheye_ids]

        if not self.return_fisheye:
            fisheye_ids, img_ids_fish = [], []

        if self.color_aug:
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
        imgs_p_left, imgs_f_left, imgs_p_right, imgs_f_right = self.load_images(
            sequence, img_ids, load_left, load_right, img_ids_fish=img_ids_fish
        )

        _loading_time = np.array(time.time() - _start_time_loading)

        _start_time_processing = time.time()
        resampler_02, resampler_03 = self._get_resamplers(
            self._calibs, tuple(imgs_p_left[0].shape[:2])
        )

        imgs_p_left = [
            self.process_img(img, color_aug_fn=color_aug_fn) for img in imgs_p_left
        ]
        imgs_f_left = [
            self.process_img(img, color_aug_fn=color_aug_fn, resampler=resampler_02)
            for img in imgs_f_left
        ]
        imgs_p_right = [
            self.process_img(img, color_aug_fn=color_aug_fn) for img in imgs_p_right
        ]
        imgs_f_right = [
            self.process_img(img, color_aug_fn=color_aug_fn, resampler=resampler_03)
            for img in imgs_f_right
        ]
        _processing_time = np.array(time.time() - _start_time_processing)

        # These poses are camera to world !!
        poses_p_left = (
            [
                self._poses[sequence][i, :, :] @ self._calibs["T_cam_to_pose"]["00"]
                for i in stereo_ids
            ]
            if load_left
            else []
        )
        poses_f_left = (
            [
                self._poses[sequence][i, :, :] @ self._calibs["T_cam_to_pose"]["02"]
                for i in fisheye_ids
            ]
            if load_left
            else []
        )
        poses_p_right = (
            [
                self._poses[sequence][i, :, :] @ self._calibs["T_cam_to_pose"]["01"]
                for i in stereo_ids
            ]
            if load_right
            else []
        )
        poses_f_right = (
            [
                self._poses[sequence][i, :, :] @ self._calibs["T_cam_to_pose"]["03"]
                for i in fisheye_ids
            ]
            if load_right
            else []
        )

        projs_p_left = [self._calibs["K_00"] for _ in stereo_ids] if load_left else []
        projs_f_left = [self._calibs["K_00"] for _ in fisheye_ids] if load_left else []
        projs_p_right = [self._calibs["K_01"] for _ in stereo_ids] if load_right else []
        projs_f_right = (
            [self._calibs["K_01"] for _ in fisheye_ids] if load_right else []
        )

        imgs = torch.stack(
            (
                imgs_p_left + imgs_p_right + imgs_f_left + imgs_f_right
                if not is_right
                else imgs_p_right + imgs_p_left + imgs_f_right + imgs_f_left
            ),
            dim=0,
        )
        projs = torch.from_numpy(
            np.stack(
                (
                    projs_p_left + projs_p_right + projs_f_left + projs_f_right
                    if not is_right
                    else projs_p_right + projs_p_left + projs_f_right + projs_f_left
                ),
                axis=0,
            )
        )
        poses = torch.from_numpy(
            np.stack(
                (
                    poses_p_left + poses_p_right + poses_f_left + poses_f_right
                    if not is_right
                    else poses_p_right + poses_p_left + poses_f_right + poses_f_left
                ),
                axis=0,
            )
        )

        ids = np.array(
            stereo_ids + stereo_ids + fisheye_ids + fisheye_ids, dtype=np.int32
        )

        src_ids = self.encoder_sampling(len(ids))
        src_frames = {
            "imgs": imgs[src_ids],
            "projs": projs[src_ids],
            "poses": poses[src_ids],
            "ts": [ids[idx] for idx in src_ids],
        }

        ref_frames = {
            "imgs": imgs,
            "projs": projs,
            "poses": poses,
            "ts": ids,
        }

        # TODO: Fix this so it loads the depth for the correct src_frame
        if self.return_depth:
            depths = torch.from_numpy(
                np.stack(
                    [
                        self.load_depth(
                            sequence,
                            img_ids[idx],
                            is_right,
                            src_frames["imgs"].shape[-2:],
                        )
                        for idx in src_ids
                    ]
                )
            )
            src_frames["depths"] = depths

        if self.return_segmentation:
            seg_p_left, seg_p_right, seg_f_left, seg_f_right = (
                self.load_segmentation_images(
                    sequence, img_ids, load_left, load_right, img_ids_fish=img_ids_fish
                )
            )
            segs = torch.stack(
                (
                    seg_p_left + seg_p_right + seg_f_left + seg_f_right
                    if not is_right
                    else seg_p_right + seg_p_left + seg_f_right + seg_f_left
                ),
                dim=0,
            )
            ref_frames["segs"] = segs

        # if self.return_segmentation:
        #     segs = [self.load_segmentation(sequence, img_ids[0])]
        # else:
        #     segs = []

        _proc_time = np.array(time.time() - _start_time)

        # print(_loading_time, _processing_time, _proc_time)

        data = {
            "src": src_frames,
            "ref": ref_frames,
            "t__get_item__": np.array([_proc_time]),
            "index": np.array([index]),
        }

        return data

    def __len__(self) -> int:
        return self.length
