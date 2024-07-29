# from ast import main
# from hmac import new
# from arrow import get
from pathlib import Path
from torch.utils.data import Dataset
from typing import get_args, get_origin
import torch
from typing import (
    Any,
    List,
    cast,
    IO,
    Optional,
    Type,
    TypeVar,
    Union,
)
import dataclasses
import gzip
from dataclasses import dataclass, Field, MISSING
import json
import time
import torchvision.transforms.functional as F
from torchvision.transforms import ColorJitter
import cv2
import numpy as np
import os
from bts.common.point_sampling import regular_grid
from dotdict import dotdict
from PIL import Image

from collections import defaultdict

from bts.common.cameras.pinhole import (
    center_crop,
    resize_to_canonical_frame,
    unnormalize_camera_intrinsics,
)
from bts.common.sampling_strategies import get_encoder_sampling

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


_X = TypeVar("_X")

TF3 = 'tuple[float, float, float]'


@dataclass
class ImageAnnotation:
    # path to jpg file, relative w.r.t. dataset_root
    path: str
    # H x W
    size: 'tuple[int, int]'  # TODO: rename size_hw?


@dataclass
class DepthAnnotation:
    # path to png file, relative w.r.t. dataset_root, storing `depth / scale_adjustment`
    path: str
    # a factor to convert png values to actual depth: `depth = png * scale_adjustment`
    scale_adjustment: float
    # path to png file, relative w.r.t. dataset_root, storing binary `depth` mask
    mask_path: Optional[str]


@dataclass
class MaskAnnotation:
    # path to png file storing (Prob(fg | pixel) * 255)
    path: str
    # (soft) number of pixels in the mask; sum(Prob(fg | pixel))
    mass: Optional[float] = None


@dataclass
class ViewpointAnnotation:
    # In right-multiply (PyTorch3D) format. X_cam = X_world @ R + T
    R: 'tuple[TF3, TF3, TF3]'
    T: TF3

    focal_length: 'tuple[float, float]'
    principal_point: 'tuple[float, float]'

    intrinsics_format: str = "ndc_norm_image_bounds"
    # Defines the co-ordinate system where focal_length and principal_point live.
    # Possible values: ndc_isotropic | ndc_norm_image_bounds (default)
    # ndc_norm_image_bounds: legacy PyTorch3D NDC format, where image boundaries
    #     correspond to [-1, 1] x [-1, 1], and the scale along x and y may differ
    # ndc_isotropic: PyTorch3D 0.5+ NDC convention where the shorter side has
    #     the range [-1, 1], and the longer one has the range [-s, s]; s >= 1,
    #     where s is the aspect ratio. The scale is same along x and y.


@dataclass
class FrameAnnotation:
    """A dataclass used to load annotations from json."""

    # can be used to join with `SequenceAnnotation`
    sequence_name: str
    # 0-based, continuous frame number within sequence
    frame_number: int
    # timestamp in seconds from the video start
    frame_timestamp: float

    image: ImageAnnotation
    depth: Optional[DepthAnnotation] = None
    mask: Optional[MaskAnnotation] = None
    viewpoint: Optional[ViewpointAnnotation] = None
    meta: Optional['dict[str, Any]'] = None


@dataclass
class PointCloudAnnotation:
    # path to ply file with points only, relative w.r.t. dataset_root
    path: str
    # the bigger the better
    quality_score: float
    n_points: Optional[int]


@dataclass
class VideoAnnotation:
    # path to the original video file, relative w.r.t. dataset_root
    path: str
    # length of the video in seconds
    length: float


@dataclass
class SequenceAnnotation:
    sequence_name: str
    category: str
    video: Optional[VideoAnnotation] = None
    point_cloud: Optional[PointCloudAnnotation] = None
    # the bigger the better
    viewpoint_quality_score: Optional[float] = None


def dump_dataclass(obj: Any, f: IO, binary: bool = False) -> None:
    """
    Args:
        f: Either a path to a file, or a file opened for writing.
        obj: A @dataclass or collection hierarchy including dataclasses.
        binary: Set to True if `f` is a file handle, else False.
    """
    if binary:
        f.write(json.dumps(_asdict_rec(obj)).encode("utf8"))
    else:
        json.dump(_asdict_rec(obj), f)


def load_dataclass(f: IO, cls: Type[_X], binary: bool = False) -> _X:
    """
    Loads to a @dataclass or collection hierarchy including dataclasses
    from a json recursively.
    Call it like load_dataclass(f, typing.List[FrameAnnotationAnnotation]).
    raises KeyError if json has keys not mapping to the dataclass fields.

    Args:
        f: Either a path to a file, or a file opened for writing.
        cls: The class of the loaded dataclass.
        binary: Set to True if `f` is a file handle, else False.
    """
    if binary:
        asdict = json.loads(f.read().decode("utf8"))
    else:
        asdict = json.load(f)

    if isinstance(asdict, list):
        # in the list case, run a faster "vectorized" version
        cls = get_args(cls)[0]
        res = list(_dataclass_list_from_dict_list(asdict, cls))
    else:
        res = _dataclass_from_dict(asdict, cls)

    return res


def _dataclass_list_from_dict_list(dlist, typeannot):
    """
    Vectorised version of `_dataclass_from_dict`.
    The output should be equivalent to
    `[_dataclass_from_dict(d, typeannot) for d in dlist]`.

    Args:
        dlist: list of objects to convert.
        typeannot: type of each of those objects.
    Returns:
        iterator or list over converted objects of the same length as `dlist`.

    Raises:
        ValueError: it assumes the objects have None's in consistent places across
            objects, otherwise it would ignore some values. This generally holds for
            auto-generated annotations, but otherwise use `_dataclass_from_dict`.
    """

    cls = get_origin(typeannot) or typeannot

    if typeannot is Any:
        return dlist
    if all(obj is None for obj in dlist):  # 1st recursion base: all None nodes
        return dlist
    if any(obj is None for obj in dlist):
        # filter out Nones and recurse on the resulting list
        idx_notnone = [(i, obj) for i, obj in enumerate(dlist) if obj is not None]
        idx, notnone = zip(*idx_notnone)
        converted = _dataclass_list_from_dict_list(notnone, typeannot)
        res = [None] * len(dlist)
        for i, obj in zip(idx, converted):
            res[i] = obj
        return res

    is_optional, contained_type = _resolve_optional(typeannot)
    if is_optional:
        return _dataclass_list_from_dict_list(dlist, contained_type)

    # otherwise, we dispatch by the type of the provided annotation to convert to
    if issubclass(cls, tuple) and hasattr(cls, "_fields"):  # namedtuple
        # For namedtuple, call the function recursively on the lists of corresponding keys
        types = cls._field_types.values()
        dlist_T = zip(*dlist)
        res_T = [
            _dataclass_list_from_dict_list(key_list, tp)
            for key_list, tp in zip(dlist_T, types)
        ]
        return [cls(*converted_as_tuple) for converted_as_tuple in zip(*res_T)]
    elif issubclass(cls, (list, tuple)):
        # For list/tuple, call the function recursively on the lists of corresponding positions
        types = get_args(typeannot)
        if len(types) == 1:  # probably List; replicate for all items
            types = types * len(dlist[0])
        dlist_T = zip(*dlist)
        res_T = (
            _dataclass_list_from_dict_list(pos_list, tp)
            for pos_list, tp in zip(dlist_T, types)
        )
        if issubclass(cls, tuple):
            return list(zip(*res_T))
        else:
            return [cls(converted_as_tuple) for converted_as_tuple in zip(*res_T)]
    elif issubclass(cls, dict):
        # For the dictionary, call the function recursively on concatenated keys and vertices
        key_t, val_t = get_args(typeannot)
        all_keys_res = _dataclass_list_from_dict_list(
            [k for obj in dlist for k in obj.keys()], key_t
        )
        all_vals_res = _dataclass_list_from_dict_list(
            [k for obj in dlist for k in obj.values()], val_t
        )
        indices = np.cumsum([len(obj) for obj in dlist])
        assert indices[-1] == len(all_keys_res)

        keys = np.split(list(all_keys_res), indices[:-1])
        # vals = np.split(all_vals_res, indices[:-1])
        all_vals_res_iter = iter(all_vals_res)
        return [cls(zip(k, all_vals_res_iter)) for k in keys]
    elif not dataclasses.is_dataclass(typeannot):
        return dlist

    # dataclass node: 2nd recursion base; call the function recursively on the lists
    # of the corresponding fields
    assert dataclasses.is_dataclass(cls)
    fieldtypes = {
        f.name: (_unwrap_type(f.type), _get_dataclass_field_default(f))
        for f in dataclasses.fields(typeannot)
    }

    # NOTE the default object is shared here
    key_lists = (
        _dataclass_list_from_dict_list([obj.get(k, default) for obj in dlist], type_)
        for k, (type_, default) in fieldtypes.items()
    )
    transposed = zip(*key_lists)
    return [cls(*vals_as_tuple) for vals_as_tuple in transposed]


def _dataclass_from_dict(d, typeannot):
    if d is None or typeannot is Any:
        return d
    is_optional, contained_type = _resolve_optional(typeannot)
    if is_optional:
        # an Optional not set to None, just use the contents of the Optional.
        return _dataclass_from_dict(d, contained_type)

    cls = get_origin(typeannot) or typeannot
    if issubclass(cls, tuple) and hasattr(cls, "_fields"):  # namedtuple
        types = cls._field_types.values()
        return cls(*[_dataclass_from_dict(v, tp) for v, tp in zip(d, types)])
    elif issubclass(cls, (list, tuple)):
        types = get_args(typeannot)
        if len(types) == 1:  # probably List; replicate for all items
            types = types * len(d)
        return cls(_dataclass_from_dict(v, tp) for v, tp in zip(d, types))
    elif issubclass(cls, dict):
        key_t, val_t = get_args(typeannot)
        return cls(
            (_dataclass_from_dict(k, key_t), _dataclass_from_dict(v, val_t))
            for k, v in d.items()
        )
    elif not dataclasses.is_dataclass(typeannot):
        return d

    assert dataclasses.is_dataclass(cls)
    fieldtypes = {f.name: _unwrap_type(f.type) for f in dataclasses.fields(typeannot)}
    return cls(**{k: _dataclass_from_dict(v, fieldtypes[k]) for k, v in d.items()})


def _unwrap_type(tp):
    # strips Optional wrapper, if any
    if get_origin(tp) is Union:
        args = get_args(tp)
        if len(args) == 2 and any(a is type(None) for a in args):  # noqa: E721
            # this is typing.Optional
            return args[0] if args[1] is type(None) else args[1]  # noqa: E721
    return tp


def _get_dataclass_field_default(field: Field) -> Any:
    if field.default_factory is not MISSING:
        # pyre-fixme[29]: `Union[dataclasses._MISSING_TYPE,
        #  dataclasses._DefaultFactory[typing.Any]]` is not a function.
        return field.default_factory()
    elif field.default is not MISSING:
        return field.default
    else:
        return None


def _asdict_rec(obj):
    return dataclasses._asdict_inner(obj, dict)


def dump_dataclass_jgzip(outfile: str, obj: Any) -> None:
    """
    Dumps obj to a gzipped json outfile.

    Args:
        obj: A @dataclass or collection hiererchy including dataclasses.
        outfile: The path to the output file.
    """
    with gzip.GzipFile(outfile, "wb") as f:
        dump_dataclass(obj, cast(IO, f), binary=True)


def load_dataclass_jgzip(outfile, cls):
    """
    Loads a dataclass from a gzipped json outfile.

    Args:
        outfile: The path to the loaded file.
        cls: The type annotation of the loaded dataclass.

    Returns:
        loaded_dataclass: The loaded dataclass.
    """
    with gzip.GzipFile(outfile, "rb") as f:
        return load_dataclass(cast(IO, f), cls, binary=True)


def _resolve_optional(type_: Any) -> 'tuple[bool, Any]':
    """Check whether `type_` is equivalent to `typing.Optional[T]` for some T."""
    if get_origin(type_) is Union:
        args = get_args(type_)
        if len(args) == 2 and args[1] == type(None):  # noqa E721
            return True, args[0]
    if type_ is Any:
        return True, Any

    return False, type_


SEQUENCE_NAME = "268_28455_57205"


class CO3D_Dataset(Dataset):
    def __init__(
        self,
        data_path: str = "/storage/group/dataset_mirrors/01_incoming/CO3D/data",
        scale_alignment_file: Path | None = None,
        category_names: List[str] = ["hydrant"],
        train: bool = True,
        known: bool = True,
        color_augmentation: bool = True,
        target_image_size: 'tuple[int, int]' = (640, 360),  # H, W
        n_image_pairs: int = 1,
        max_dist_between_frames_in_pair: int = 5,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.scale_alignment_file = scale_alignment_file
        self.category_names = category_names
        self.train = train
        self.known = known
        self.color_augmentation = color_augmentation
        self.target_image_size = target_image_size
        self.n_image_pairs = n_image_pairs
        self.max_dist_between_frames_in_pair = max_dist_between_frames_in_pair
        if self.scale_alignment_file is not None:
            self.scales = self.get_scale_alignment()
        else:
            self.scales = None
        (
            self.frame_annotations,
            self.set_lists,
        ) = self.get_frame_annotations_by_categories()
        # (
        #     self.frame_annotations,
        #     self.set_lists,
        # ) = self.setup_frame_annotations()
        self.length = sum(
            [
                len(self.set_lists[category_name])
                for category_name in self.category_names
            ]
        )
        self.canonical_K = torch.Tensor(
            [
                [500.0, 0, 0.0],
                [0, 500.0, 0.0],
                [0, 0, 1],
            ]
        )

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
            X_RANGE=(-3, 3),
            Y_RANGE=(0.0, 0.75),
            Z_RANGE=(9, 3),
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

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        _start_time = time.time()
        if index >= len(self):
            raise IndexError("Index out of bounds")
        # (category, sequence, frame_indices) = self.get_locations_of_frames(
        #     index, only_neighbouring_frames=not self.train
        # )
        (category, sequence, frame_indices) = self.get_locations_of_frames(
            index, only_neighbouring_frames=True
        )
        if self.scales is not None:
            scale = self.scales.get(category, {}).get(sequence, 1.0)
        else:
            scale = 1.0

        if self.color_augmentation:
            color_aug_fn = self.get_color_aug_fn(
                ColorJitter.get_params(
                    brightness=[0.8, 1.2],
                    contrast=[0.8, 1.2],
                    saturation=[0.8, 1.2],
                    hue=[-0.1, 0.1],
                )
            )
        else:
            color_aug_fn = None
        # get necessary frame information
        processed_images = []
        calibrations = []
        poses = []
        depths = []
        for frame_index in frame_indices:
            frame_annotation = self.frame_annotations[category][sequence][frame_index]
            img = torch.from_numpy(
                cv2.cvtColor(
                    cv2.imread(
                        os.path.join(
                            self.data_path,
                            frame_annotation.image.path,
                        )
                    ),
                    cv2.COLOR_BGR2RGB,
                ).astype(np.float32)
                / 255
            )
            K = self.getCalibs(frame_annotation)

            with Image.open(
                os.path.join(
                    self.data_path,
                    frame_annotation.depth.path,
                )
            ) as depth_pil:
                depth = (
                    np.frombuffer(
                        np.array(depth_pil, dtype=np.uint16), dtype=np.float16
                    )
                    .astype(np.float32)
                    .reshape((depth_pil.size[1], depth_pil.size[0]))
                ) * frame_annotation.depth.scale_adjustment
                depth[~np.isfinite(depth)] = 0.0

            depth = torch.from_numpy(depth)[None] * scale

            img, K, depth = self.process_frame(
                img,
                K,
                color_aug_fn=color_aug_fn,
                depth_map=depth,
            )
            offset = np.eye(4, dtype=np.float32)
            offset[0, 0] = -1.0
            offset[1, 1] = -1.0
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = np.asarray(frame_annotation.viewpoint.R).T
            pose[:3, 3] = frame_annotation.viewpoint.T
            pose[:3, 3] = pose[:3, 3] * scale
            pose = np.linalg.inv(offset @ pose)

            # Add information of frame to current pair
            processed_images.append(img)
            # K[:2, :] = -K[:2, :]
            calibrations.append(K)
            poses.append(torch.from_numpy(pose))
            depths.append(depth)

        processed_images = torch.stack(processed_images)
        calibrations = torch.stack(calibrations)
        poses = torch.stack(poses)
        depths = torch.stack(depths)

        src_ids = self.encoder_sampling(processed_images.shape[0])
        ref_ids = [idx for idx in range(processed_images.shape[0])]
        # ref_ids = [
        #     idx for idx in range(processed_images.shape[0]) if not idx in src_ids
        # ]

        src_frames = {
            "imgs": processed_images[src_ids],
            "poses": poses[src_ids],
            "projs": calibrations[src_ids],
            "depths": depths[src_ids],
        }

        ref_frames = {
            "imgs": processed_images[ref_ids],
            "poses": poses[ref_ids],
            "projs": calibrations[ref_ids],
            "depths": depths,
        }

        _proc_time = time.time() - _start_time

        return {
            "src": src_frames,
            "ref": ref_frames,
            "t__get_item__": _proc_time,
        }

    def get_scale_alignment(self) -> 'dict[str, dict[str, float]]':
        with open(self.scale_alignment_file, "r") as f:
            scale_alignment = json.load(f)

        def filter(
            mean: torch.Tensor, std: torch.Tensor, norm_std: torch.Tensor
        ) -> float:

            return 1 / torch.mean(mean[norm_std < 0.2]).nan_to_num(mean.mean()).item()

        filtered_scale_alignment = {}
        for category, sequences in scale_alignment.items():
            filtered_scale_alignment[category] = {}
            for sequence, scales in sequences.items():
                mean, std, norm_std = [], [], []
                for scale in scales:
                    mean.append(scale[0])
                    std.append(scale[1])
                    norm_std.append(scale[2])
                filtered_scale_alignment[category][sequence] = filter(
                    torch.tensor(mean), torch.tensor(std), torch.tensor(norm_std)
                )

        return filtered_scale_alignment

    # return  category_name, sequence_name, main_frame_index, [frame_pair_indices: tuple[int, int]]
    def get_locations_of_frames(
        self, index: int, only_neighbouring_frames=True
    ) -> 'tuple[str, str, List[int]]':
        for category_name in self.category_names:
            if index < len(self.set_lists[category_name]):
                sequence_name = self.set_lists[category_name][index][0]
                # index inside of sequence:
                beginning_of_sequence_index = index
                while (
                    self.set_lists[category_name][beginning_of_sequence_index][0]
                    == sequence_name
                    and beginning_of_sequence_index > 0
                ):
                    beginning_of_sequence_index -= 1
                frame_indices: List[int] = []
                main_frame_index = (
                    index - beginning_of_sequence_index - 1
                    if beginning_of_sequence_index > 0
                    else index
                )
                frame_indices.append(main_frame_index)
                sequence_size = len(
                    self.frame_annotations[category_name][sequence_name]
                )
                if only_neighbouring_frames:
                    # for testing: returns 6 neighbouring frames
                    for i in range(0, 4):
                        if i != 0:
                            if main_frame_index + i < 0:
                                frame_indices.append(main_frame_index + 7 + i)
                            elif main_frame_index + i >= sequence_size:
                                frame_indices.append(main_frame_index - 7 + i)
                            else:
                                frame_indices.append(main_frame_index + i)
                else:
                    frame_indices.append(
                        self.get_random_index_in_neighborhood(
                            main_frame_index,
                            sequence_size,
                            self.max_dist_between_frames_in_pair,
                        )
                    )
                    # find additional frame pairs based on main frame
                    # always add frame pair at the opposite side of the main frame
                    opposite_frame_index = (
                        main_frame_index + sequence_size // 2
                        if main_frame_index + sequence_size // 2 < sequence_size
                        else main_frame_index - sequence_size // 2
                    )
                    frame_indices.append(opposite_frame_index)
                    frame_indices.append(
                        self.get_random_index_in_neighborhood(
                            opposite_frame_index,
                            sequence_size,
                            self.max_dist_between_frames_in_pair,
                        )
                    )
                    # add additional frame pairs randomly
                    for _ in range(1, self.n_image_pairs):
                        new_frame_index = np.random.randint(0, sequence_size - 1)
                        if new_frame_index == main_frame_index:
                            new_frame_index = sequence_size - 1
                        frame_indices.append(new_frame_index)
                        frame_indices.append(
                            self.get_random_index_in_neighborhood(
                                new_frame_index,
                                sequence_size,
                                self.max_dist_between_frames_in_pair,
                            )
                        )
                return (category_name, sequence_name, frame_indices)

            else:
                index -= len(self.set_lists[category_name])
        raise IndexError("Index out of bounds")

    @staticmethod
    def get_random_index_in_neighborhood(
        base_index: int, interval_size: int, max_diff: int
    ) -> int:
        neighbor = base_index
        while neighbor == base_index:
            neighbor = np.random.randint(
                max(0, base_index - max_diff),
                min(base_index + max_diff, interval_size - 1),
            )
        return neighbor

    def getCalibs(self, frame_annotation: FrameAnnotation) -> torch.Tensor:
        image_size = frame_annotation.image.size
        f = (
            frame_annotation.viewpoint.focal_length
            if frame_annotation.viewpoint
            else (0, 0)
        )
        p = (
            frame_annotation.viewpoint.principal_point
            if frame_annotation.viewpoint
            else (0, 0)
        )
        calibration_matrix = torch.eye(3)
        calibration_matrix[0, 0] = f[0]
        calibration_matrix[1, 1] = f[1]
        calibration_matrix[0, 2] = p[0]
        calibration_matrix[1, 2] = p[1]

        calibration_matrix = unnormalize_camera_intrinsics(
            calibration_matrix[None], torch.tensor(image_size)
        )[0]

        return calibration_matrix

    def get_frame_annotations_by_categories(self) -> 'tuple[Any, Any]':
        unfiltered_frame_annotations = defaultdict(lambda: defaultdict(defaultdict))
        frame_annotations = defaultdict(lambda: defaultdict(list))
        set_lists = {}
        for category_name in self.category_names:
            try:
                with open(f"{self.data_path}/{category_name}/set_lists.json", "r") as f:
                    set_list = json.load(f)[
                        f"{'train' if self.train else 'test'}_{'known' if self.known else 'unseen'}"
                    ]
                    set_lists[category_name] = set_list
            except FileNotFoundError:
                print(f"File not found for category {category_name}")

            # load frame annotations
            all_frame_annotations = load_dataclass_jgzip(
                f"{self.data_path}/{category_name}/frame_annotations.jgz",
                List[FrameAnnotation],
            )

            # rearrange nested dict: frame_annotations[category_name][sequence_name][frame_index] for faster access on individual elements
            # but only add frame annotations that are in respective set_list
            for frame_annotation in all_frame_annotations:
                unfiltered_frame_annotations[category_name][
                    frame_annotation.sequence_name
                ][frame_annotation.frame_number] = frame_annotation

            for element in set_lists[category_name]:
                sequence_name = element[0]
                frame_number = element[1]
                frame_annotations[category_name][sequence_name].append(
                    unfiltered_frame_annotations[category_name][sequence_name][
                        frame_number
                    ]
                )

        return (frame_annotations, set_lists)

    def setup_frame_annotations(self) -> List:
        unfiltered_frame_annotations = {}
        frame_annotations = []
        unfiltered_set_list = []
        set_list = []
        try:
            with open(f"{self.data_path}/hydrant/set_lists.json", "r") as f:
                unfiltered_set_list = json.load(f)[
                    f"{'train' if self.train else 'test'}_{'known' if self.known else 'unseen'}"
                ]
        except FileNotFoundError:
            print(f"File not found")

        for element in unfiltered_set_list:
            if element[0] == SEQUENCE_NAME:
                set_list.append(element)

        # load frame annotations
        all_frame_annotations = load_dataclass_jgzip(
            f"{self.data_path}/hydrant/frame_annotations.jgz",
            List[FrameAnnotation],
        )
        for frame_annotation in all_frame_annotations:
            if frame_annotation.sequence_name == SEQUENCE_NAME:
                unfiltered_frame_annotations[frame_annotation.frame_number] = (
                    frame_annotation
                )

        for element in set_list:
            frame_number = element[1]
            frame_annotations.append(unfiltered_frame_annotations[frame_number])

        return {"hydrant": {SEQUENCE_NAME: frame_annotations}}, {"hydrant": set_list}

    # apply color augmentation and resizing to single image
    # img: H, W, C
    def process_frame(
        self,
        img: torch.Tensor,
        K: torch.Tensor,
        color_aug_fn,
        depth_map: torch.Tensor | None = None,
    ):
        # TODO: canonicalize image intrisics
        # TODO: crop image to common aspect ratio

        img = img.permute(2, 0, 1)

        if depth_map is not None:
            img, K, depth_map = resize_to_canonical_frame(
                img[None], K[None], self.canonical_K, depth_map[None]
            )
            img, K, depth_map = center_crop(img, K, self.target_image_size, depth_map)
            img = img[0]
            K = K[0]
            depth_map = depth_map[0]

        else:
            img, K = resize_to_canonical_frame(img[None], K[None], self.canonical_K)
            img, K = center_crop(img, K, self.target_image_size)
            img = img[0]
            depth_map = depth_map[0]

        if color_aug_fn is not None:
            img = color_aug_fn(img)
        return img, K, depth_map

    @staticmethod
    def get_color_aug_fn(
        params: 'tuple[torch.Tensor, float | None, float | None, float | None, float | None]',
    ):
        (
            fn_idx,
            brightness_factor,
            contrast_factor,
            saturation_factor,
            hue_factor,
        ) = params

        def color_aug_fn(img: torch.Tensor):
            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    img = F.adjust_brightness(img, brightness_factor)
                elif fn_id == 1 and contrast_factor is not None:
                    img = F.adjust_contrast(img, contrast_factor)
                elif fn_id == 2 and saturation_factor is not None:
                    img = F.adjust_saturation(img, saturation_factor)
                elif fn_id == 3 and hue_factor is not None:
                    img = F.adjust_hue(img, hue_factor)

            return img

        return color_aug_fn
