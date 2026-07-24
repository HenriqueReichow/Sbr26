"""CSV logging for the onboard-perception gesture runs (the paper's results table).

Three files per run under <out_dir>/<run_id>/:

  gestures.csv  one row per processed camera frame
  landings.csv  one row per landing outcome on a Phase-3 platform
  mission.csv   exactly one row: the per-run mission outcome metrics

Every gestures.csv row carries what the analysis needs and nothing it doesn't:

  motion_condition     0 static / 1 altitude-only / 2 full closed-loop. The independent
                       variable: how much of the recognized command the drone executed.
  run_id, operator_id  set at launch (--run-id / --operator-id)
  instructed_command   ground truth: the command the operator was told to give. Empty
                       while the pose is still settling after a change -- those frames
                       have no ground truth and are excluded from scoring.
  recognized_command   what the drone recognized from its own camera, post-debounce
  raw_command          pre-debounce, so the debounce ablation is inspectable
  capture_timestamp    when the frame was pulled off the simulated camera
  command_timestamp    when the resulting command existed
  inference_time_ms    pose estimation + command inference for THAT frame, on the
                       recognition host. Excludes simulator stepping and rendering.
  mission_phase        takeoff | gesture | land | done. Only `gesture` frames carry an
                       instructed_command, so only they are scored for recognition.
  drone_x/y/z          BiguaSim ground-truth position at that camera frame
  drone_yaw_deg        BiguaSim ground-truth heading
  operator_detected    1 if the operator was found in the drone's camera this frame.
                       NOT the same as raw_command != "none": a body squarely in frame
                       with arms in no commanded pose is detected but classifies `none`.
  loss_reason          "" when detected, else no_landmarks (out of frame entirely) |
                       low_visibility (found, but partly cut off / occluded) |
                       small_torso (in frame, too far away). Kept separable because only
                       no_landmarks is an outright framing loss.
  torso_height         normalised shoulder-to-hip extent; the classifier's distance gate
  operator_root_yaw_deg   the AVATAR's facing, world degrees
  drone_bearing_deg       bearing from the operator to the drone, world degrees
  view_angle_deg          drone_bearing - operator_root_yaw, wrapped to [-180, 180].
                       0 = the drone sees a frontal body; +/-90 = side-on; +/-180 = back.
                       This is what makes "operator turned away" separable from
                       "operator left the frame" in the data.

mission.csv holds the landing-task outcome, one row per run:

  tracking_loss_fraction  fraction of the mission's camera frames with no operator
                       detected -- the study's primary metric. Empty (not 0.0) when the
                       run processed no camera frames.
  frames_total, frames_operator_lost   the numerator and denominator, so the fraction is
                       auditable and runs can be pooled by frame count rather than by run.
  operator_facing_mode  face_drone | fixed

  landing_success      did the drone come to rest on the correct base (bool)
  landing_error_m      HORIZONTAL distance from the final position to the base centre.
                       Horizontal, not 3D: a 3D error would be floored near land_z_max
                       by the descent cutoff and would describe the cutoff, not the
                       landing precision.
  time_to_land_s       SIMULATED seconds from the start of the land phase to touchdown
                       (ticks / ticks_per_sec). Deterministic; independent of CPU load.
  path_length_m        ground-truth path length integrated over the whole mission

Summarise with `python tools/analyze_runs.py logs/`.
"""
from __future__ import annotations

import csv
import os
from typing import Sequence

GESTURE_FIELDS = [
    "run_id", "operator_id", "mode", "motion_condition", "lighting",
    "frame_id", "capture_timestamp", "command_timestamp",
    "instructed_command", "raw_command", "recognized_command",
    "command_sent_to_drone", "inference_time_ms",
    "mission_phase", "drone_x", "drone_y", "drone_z", "drone_yaw_deg",
    "operator_detected", "loss_reason", "torso_height",
    "operator_root_yaw_deg", "drone_bearing_deg", "view_angle_deg",
    "yaw_mode", "bearing_error_deg", "last_seen_bearing_deg", "heading_error_gt_deg",
]
LANDING_FIELDS = ["run_id", "timestamp", "platform", "result", "pos_x", "pos_y", "pos_z"]
MISSION_FIELDS = [
    "run_id", "operator_id", "mode", "motion_condition", "lighting", "seed",
    "target_base", "landing_success", "landing_error_m", "time_to_land_s",
    "path_length_m", "final_x", "final_y", "final_z", "outcome",
    "camera_hfov_calibrated",
    "tracking_loss_fraction", "frames_total", "frames_operator_lost",
    "operator_facing_mode", "yaw_mode", "operator_mode",
    "loss_share_no_landmarks", "loss_share_low_visibility", "loss_share_small_torso",
    "loss_recovery",
]


def _fmt_cmd(cmd: Sequence[float]) -> str:
    """cmd_vel as a single stable CSV cell, e.g. '0.000;0.000;0.600'."""
    return ";".join(f"{v:.3f}" for v in cmd)


class GestureLogger:
    def __init__(self, meta: dict, out_dir: str):
        self.meta = meta
        self.run_dir = os.path.join(out_dir, str(meta["run_id"]))
        os.makedirs(self.run_dir, exist_ok=True)
        self._g_f, self._g = self._open("gestures.csv", GESTURE_FIELDS)
        self._l_f, self._l = self._open("landings.csv", LANDING_FIELDS)

    def _open(self, name, fields):
        f = open(os.path.join(self.run_dir, name), "w", newline="")
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        return f, w

    def log_gesture(self, frame_id, raw, recognized, cmd, capture_ts, command_ts,
                    inference_s, instructed="", phase="", pose=None, frame_meta=None):
        pos, yaw = (pose[0], pose[1][2]) if pose is not None else ((None,) * 3, None)
        meta = frame_meta or {}
        unknown = set(meta) - set(GESTURE_FIELDS)
        if unknown:
            raise ValueError(f"unknown gesture fields: {sorted(unknown)}")
        self._g.writerow({
            **meta,
            "run_id": self.meta["run_id"],
            "operator_id": self.meta["operator_id"],
            "mode": self.meta["mode"],
            "motion_condition": self.meta["motion_condition"],
            "lighting": self.meta["lighting"],
            "frame_id": frame_id,
            "capture_timestamp": f"{capture_ts:.6f}",
            "command_timestamp": f"{command_ts:.6f}",
            "instructed_command": instructed,
            "raw_command": raw,
            "recognized_command": recognized,
            "command_sent_to_drone": _fmt_cmd(cmd),
            "inference_time_ms": f"{inference_s * 1000.0:.3f}",
            "mission_phase": phase,
            "drone_x": "" if pos[0] is None else f"{pos[0]:.4f}",
            "drone_y": "" if pos[1] is None else f"{pos[1]:.4f}",
            "drone_z": "" if pos[2] is None else f"{pos[2]:.4f}",
            "drone_yaw_deg": "" if yaw is None else f"{yaw:.2f}",
        })
        self._g_f.flush()

    def log_landing(self, timestamp, platform, result, pos):
        self._l.writerow({
            "run_id": self.meta["run_id"],
            "timestamp": f"{timestamp:.6f}",
            "platform": platform,
            "result": result,
            "pos_x": f"{pos[0]:.4f}", "pos_y": f"{pos[1]:.4f}", "pos_z": f"{pos[2]:.4f}",
        })
        self._l_f.flush()

    def log_mission(self, **row) -> None:
        """The run's single mission.csv row. Written once, at the end of the mission."""
        unknown = set(row) - set(MISSION_FIELDS)
        if unknown:
            raise ValueError(f"unknown mission fields: {sorted(unknown)}")
        with open(os.path.join(self.run_dir, "mission.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=MISSION_FIELDS)
            w.writeheader()
            w.writerow({k: row.get(k, "") for k in MISSION_FIELDS})

    def close(self) -> None:
        self._g_f.close()
        self._l_f.close()
