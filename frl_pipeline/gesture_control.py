"""Command -> cmd_vel controller. The single decision point of the pipeline.

    landmarks --classify--> raw --debounce--> detected --map--> cmd_vel [vx, vy, vz]

(`classify` runs in the caller so that it can be timed together with pose estimation.)

Full mode and the baseline ablation share EVERY step here; they differ only in the
debounce horizon H, which `mode_hold_frames()` resolves. That keeps the comparison
an honest ablation of this pipeline rather than a strawman (paper Sec. V-B).

The last detected command latches: it keeps driving the drone until a different one
is detected. `land` additionally scores a landing when the drone is over a platform.
"""
from __future__ import annotations

import math
from typing import Callable

from .gestures import NONE, Debouncer


def mode_hold_frames(cfg: dict) -> int:
    """H for the active mode. The baseline ablation is 'no debounce', i.e. H=1."""
    mode = cfg["run"]["mode"]
    if mode == "full":
        return int(cfg["gestures"]["H"])
    if mode == "baseline_debounce_off":
        return 1
    raise ValueError(f"unknown mode: {mode!r} (expected full | baseline_debounce_off)")


class GestureController:
    def __init__(self, cfg: dict, logger, clock: Callable[[], float]):
        self.cfg = cfg
        self.logger = logger
        self.clock = clock

        self._debouncer = Debouncer(mode_hold_frames(cfg))
        self._cmd_table = cfg["gesture_command_map"]

        pf = cfg["world"]["platforms"]
        self._platforms = pf["bases"]
        self._landing_radius = pf["landing_radius"]
        self._land_z_max = pf["land_z_max"]
        self._landed = set()

        self._active = NONE

    @property
    def active_gesture(self) -> str:
        return self._active

    def reset(self) -> None:
        self._active = NONE

    def update(self, raw: str, pose, frame_id: int, capture_ts: float,
               inference_s: float, instructed: str = "", phase: str = "",
               frame_meta: dict = None) -> list:
        """Debounce one classified frame and return the cmd_vel to send. Logs one row.

        `raw` is the caller's pre-debounce `gestures.classify()` output -- the caller
        runs it so it can time pose estimation and classification together as
        `inference_s`. `capture_ts` is when the frame left the simulated camera.
        `instructed` is the command the operator was told to give, or "" for frames with
        no ground truth (e.g. mid-transition).
        """
        # The debounced command IS the active one, `none` included: dropping the arms
        # must stop the drone. Latching `none` away instead makes the last command run
        # forever, which flies the operator out of frame and starves recognition.
        self._active = self._debouncer.update(raw)
        detected = self._active

        table = self._cmd_table
        cmd = list(table.get(self._active, table["none"]))

        command_ts = self.clock()
        self.logger.log_gesture(frame_id, raw, detected, cmd, capture_ts, command_ts,
                                inference_s, instructed, phase, pose, frame_meta)

        if pose is not None:
            self._check_landing(command_ts, pose)
        return cmd

    def _check_landing(self, now: float, pose) -> None:
        """Score a landing once, when `land` is active over an unclaimed platform."""
        if self._active != "land":
            return
        (px, py, pz), _ = pose
        for base in self._platforms:
            if base["name"] in self._landed:
                continue
            bx, by, bz = base["pos"]
            if math.hypot(px - bx, py - by) <= self._landing_radius:
                result = "success" if pz <= bz + self._land_z_max else "fail"
                self.logger.log_landing(now, base["name"], result, (px, py, pz))
                self._landed.add(base["name"])
                return
