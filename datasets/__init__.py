import os
from pathlib import Path

#from datasets.co3d import CO3D_Dataset

from .base_dataset import BaseDataset
#from .kitti_360 import KITTI360Dataset

# from .old_kitti_360 import OldKITTI360Dataset
from .afm_kitti_360 import AFMKITTI360Dataset


# TODO: make more generic -> no more two function
def make_datasets(config) -> 'tuple[BaseDataset, BaseDataset]':
    dataset_type = config.get("type", "KITTI_360")
    if dataset_type == "KITTI_360":
        if config.get("split_path", None) is None:
            train_split_path = None
            test_split_path = None
        else:
            train_split_path = Path(config["split_path"]) / "train_files.txt"
            test_split_path = Path(config["split_path"]) / "val_files.txt"
        train_dataset = AFMKITTI360Dataset(
            data_path=Path(config["data_path"]),
            pose_path=Path(config["pose_path"]),
            split_path=train_split_path,
            target_image_size=tuple(config.get("image_size", (192, 640))),
            return_stereo=config.get("data_stereo", True),
            return_fisheye=config.get("data_fisheye", True),
            frame_count=config.get("data_fc", 3),
            return_depth=False,
            return_segmentation=config.get("data_segmentation", False),
            return_occupancy=False,
            keyframe_offset=config.get("keyframe_offset", 0),
            dilation=config.get("dilation", 1),
            fisheye_rotation=config.get("fisheye_rotation", 0),
            fisheye_offsets=config.get("fisheye_offset", [10]),
            stereo_offsets=config.get("stereo_offset", [1]),
            # color_aug=config.get("color_aug", False),
            is_preprocessed=config.get("is_preprocessed", False),
        )
        test_dataset = AFMKITTI360Dataset(
            data_path=Path(config["data_path"]),
            pose_path=Path(config["pose_path"]),
            split_path=test_split_path,
            target_image_size=tuple(config.get("image_size", (192, 640))),
            return_stereo=config.get("data_stereo", True),
            return_fisheye=config.get("data_fisheye", True),
            frame_count=config.get("data_fc", 3),
            return_depth=True,
            return_segmentation=config.get("data_segmentation", False),
            return_occupancy=config.get("occupancy", False),
            keyframe_offset=config.get("keyframe_offset", 0),
            dilation=config.get("dilation", 1),
            fisheye_rotation=config.get("fisheye_rotation", 0),
            fisheye_offsets=[10],
            stereo_offsets=[1],
            is_preprocessed=config.get("is_preprocessed", False),
        )
        return train_dataset, test_dataset
    elif dataset_type == "old_KITTI_360":
        if config.get("split_path", None) is None:
            train_split_path = None
            test_split_path = None
        else:
            train_split_path = Path(config["split_path"]) / "train_files.txt"
            test_split_path = Path(config["split_path"]) / "val_files.txt"
        train_dataset = AFMKITTI360Dataset(
            data_path=Path(config["data_path"]),
            pose_path=Path(config["pose_path"]),
            segmentation_path=(
                Path(config["segmentation_path"])
                if config.get("segmentation_path", None)
                else None
            ),
            split_path=train_split_path,
            target_image_size=tuple(config.get("image_size", (192, 640))),
            return_stereo=config.get("data_stereo", True),
            return_fisheye=config.get("data_fisheye", True),
            frame_count=config.get("data_fc", 3),
            return_depth=False,
            return_segmentation=config.get("data_segmentation", False),
            # keyframe_offset=config.get("keyframe_offset", 0),
            # dilation=config.get("dilation", 1),
            fisheye_rotation=config.get("fisheye_rotation", 0),
            fisheye_offset=config.get("fisheye_offset", [10]),
            pinhole_offset=config.get("stereo_offset", [1]),
            color_aug=config.get("color_aug", False),
            is_preprocessed=config.get("is_preprocessed", False),
        )
        test_dataset = AFMKITTI360Dataset(
            data_path=Path(config["data_path"]),
            pose_path=Path(config["pose_path"]),
            segmentation_path=(
                Path(config["segmentation_path"])
                if config.get("segmentation_path", None)
                else None
            ),
            split_path=test_split_path,
            target_image_size=tuple(config.get("image_size", (192, 640))),
            return_stereo=config.get("data_stereo", True),
            return_fisheye=config.get("data_fisheye", True),
            frame_count=config.get("data_fc", 3),
            return_depth=True,
            return_segmentation=config.get("data_segmentation", False),
            # keyframe_offset=config.get("keyframe_offset", 0),
            # dilation=config.get("dilation", 1),
            fisheye_rotation=config.get("fisheye_rotation", 0),
            fisheye_offset=[10],
            pinhole_offset=[1],
            is_preprocessed=config.get("is_preprocessed", False),
        )
        return train_dataset, test_dataset
    elif dataset_type == "CO3D":
        train_dataset = CO3D_Dataset(
            data_path=config["data_path"],
            scale_alignment_file=config.get("scale_alignment_file", None),
            category_names=config.get("category_names", ["hydrant"]),
            train=True,
            known=config.get("known", True),
            color_augmentation=True,
            target_image_size=config.get("target_image_size", [480, 480]),
            n_image_pairs=config.get("frame_count", 3),
            max_dist_between_frames_in_pair=config.get(
                "max_dist_between_frames_in_pair", 1
            ),
        )
        test_dataset = CO3D_Dataset(
            data_path=config["data_path"],
            scale_alignment_file=config.get("scale_alignment_file", None),
            category_names=config.get("category_names", ["hydrant"]),
            train=True,
            known=False,
            color_augmentation=False,
            target_image_size=config.get("target_image_size", [480, 480]),
            n_image_pairs=config.get("frame_count", 3),
            max_dist_between_frames_in_pair=config.get(
                "max_dist_between_frames_in_pair", 1
            ),
        )
        return train_dataset, test_dataset
    else:
        raise NotImplementedError(f"Unsupported dataset type: {type}")


def make_test_dataset(config):
    dataset_type = config.get("type", "KITTI_Raw")
    if dataset_type == "KITTI_360":
        test_dataset = AFMKITTI360Dataset(
            data_path=config["data_path"],
            pose_path=config["pose_path"],
            split_path=os.path.join(
                config.get("split_path", None), "test_files.txt"
            ),
            target_image_size=tuple(config.get("image_size", (192, 640))),
            frame_count=config.get("data_fc", 1),
            return_stereo=config.get("data_stereo", False),
            return_fisheye=config.get("data_fisheye", False),
            return_3d_bboxes=config.get("data_3d_bboxes", False),
            return_segmentation=config.get("data_segmentation", False),
            keyframe_offset=0,
            fisheye_rotation=config.get("fisheye_rotation", 0),
            fisheye_offset=config.get("fisheye_offset", 1),
            dilation=config.get("dilation", 1),
            is_preprocessed=config.get("is_preprocessed", False),
        )
        return test_dataset
    elif dataset_type == "CO3D":
        test_dataset = CO3D_Dataset(
            data_path=config["data_path"],
            category_names=config.get("category_names", ["hydrant"]),
            train=False,
            known=config.get("known", True),
            color_augmentation=False,
            target_image_size=config.get("target_image_size", 480, 480),
            n_image_pairs=config.get("frame_count", 3),
            max_dist_between_frames_in_pair=config.get(
                "max_dist_between_frames_in_pair", 1
            ),
        )
        return test_dataset
    else:
        raise NotImplementedError(f"Unsupported dataset type: {dataset_type}")
