from pathlib import Path
import numpy as np
from kiss_icp.pipeline import OdometryPipeline
from kiss_icp.metrics import absolute_trajectory_error, sequence_error

# from afm_kiss_icp.AFMVisualizer import AFMRegistrationVisualizer


# Redefine the _run_evaluation method of the OdometryPipeline to save the results
class AFMOdometryPipeline(OdometryPipeline):
    def __init__(
        self,
        dataset,
        config: Path | None = None,
        deskew: bool | None = False,
        max_range: float | None = None,
        visualize: bool = False,
        n_scans: int = -1,
        jump: int = 0,
    ):
        super().__init__(dataset, config, deskew, max_range, False, n_scans, jump)
        self.output_path = None

        if visualize:
            self.visualizer = AFMRegistrationVisualizer()

    def _run_evaluation(self):
        # Run estimation metrics evaluation, only when GT data was provided
        datapoints_ids = np.array(
            self._dataset.get_datapoints_ids()
        )  # Get the ids that relate datapoints
        gt_poses_ids = np.array(
            self._dataset.get_gt_poses_ids()
        )  # Get the ids that relate GT poses to datapoints
        ids_to_compare = np.intersect1d(datapoints_ids, gt_poses_ids)
        # Get only those poses that have a match
        datapoints_indexes_to_compare = np.in1d(
            datapoints_ids, ids_to_compare, assume_unique=True
        )
        gt_poses_indexes_to_compare = np.in1d(
            gt_poses_ids, ids_to_compare, assume_unique=True
        )
        if self.has_gt:
            avg_tra, avg_rot = sequence_error(
                self.gt_poses[gt_poses_indexes_to_compare, :, :],
                np.array(self.poses)[datapoints_indexes_to_compare, :, :],
            )
            ate_rot, ate_trans = absolute_trajectory_error(
                self.gt_poses[gt_poses_indexes_to_compare, :, :],
                np.array(self.poses)[datapoints_indexes_to_compare, :, :],
            )
            self.results.append(
                desc="Average Translation Error", units="%", value=avg_tra
            )
            self.results.append(
                desc="Average Rotational Error", units="deg/m", value=avg_rot
            )
            self.results.append(
                desc="Absolute Trajectory Error (ATE)", units="m", value=ate_trans
            )
            self.results.append(
                desc="Absolute Rotational Error (ARE)", units="rad", value=ate_rot
            )

        # Run timing metrics evaluation, always
        def _get_fps():
            total_time_s = sum(self.times) * 1e-9
            return float(len(self.times) / total_time_s)

        avg_fps = int(np.ceil(_get_fps()))
        avg_ms = int(np.ceil(1e3 * (1 / _get_fps())))
        self.results.append(
            desc="Average Frequency", units="Hz", value=avg_fps, trunc=True
        )
        self.results.append(
            desc="Average Runtime", units="ms", value=avg_ms, trunc=True
        )

