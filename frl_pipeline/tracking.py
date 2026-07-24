"""Is the operator in the drone's camera frame at all?

The drone's camera is bolted to the airframe and cannot pan or tilt, so keeping the
operator framed is the airframe's job. This module answers the prior question -- whether
the operator is visible in a given frame -- which is the primary metric of the yaw study.

It is a SECOND CONSUMER of the landmarks `sim_loop` already computes. It does not touch
`gestures.classify()`, the debouncer, or anything else on the recognition path.

Why not just use `raw_command == "none"`: that value conflates two different failures.
`classify()` returns NONE both when MediaPipe found no body at all, and when it found a
perfectly good body whose arms match no command. Only the first is a tracking loss.
`operator_detected` isolates it: the operator is detected when landmarks came back, every
needed landmark clears `min_visibility`, and the torso is large enough in frame to
classify against. Those are exactly the gates `classify()` applies before it looks at any
arm angle, so a detected frame is one where recognition had a fair chance.
"""
from __future__ import annotations

import math
from typing import Optional

# `_all_visible` is gestures.py's own visibility gate. Imported rather than reimplemented
# so that "detected" means exactly "recognition had a fair chance at this frame".
from .gestures import _all_visible, torso_height
from .perception import Landmarks


# Why a frame failed. Kept separable because they mean different things about the
# airframe: NO_LANDMARKS is the operator out of frame entirely (a framing failure, the
# thing the yaw study measures); LOW_VISIBILITY is a body found but partly occluded or
# cut off at the image edge; SMALL_TORSO is the operator in frame but too far away.
DETECTED = ""
NO_LANDMARKS = "no_landmarks"
LOW_VISIBILITY = "low_visibility"
SMALL_TORSO = "small_torso"


def bearing_error_deg(image_x: float, hfov_deg: float) -> float:
    """Yaw the drone must turn, in degrees, to centre the operator in its camera.

    Pinhole inversion of the calibration fit (tools/calibrate_camera_hfov.py):

        x_img = 0.5 + tan(theta) / (2 * tan(hfov / 2))     R^2 = 0.99999

    where theta is the drone's yaw error. The fitted slope is POSITIVE -- a drone yawed
    +theta sees the operator RIGHT of centre -- so the correction is -theta. Positive
    return means "yaw further positive (CCW)"; negative means "yaw back".
    """
    theta = math.atan((image_x - 0.5) * 2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    return -math.degrees(theta)


def _shoulder_mid_x(lm: Landmarks) -> float:
    return 0.5 * (lm["left_shoulder"][0] + lm["right_shoulder"][0])


class OperatorTrack:
    """Per-frame visibility of, and bearing to, the operator in the drone's camera."""

    __slots__ = ("detected", "torso_height", "loss_reason", "bearing_error_deg")

    def __init__(self, detected: bool, torso: float, reason: str,
                 bearing: Optional[float] = None):
        self.detected = detected
        self.torso_height = torso
        self.loss_reason = reason
        # None whenever the operator was not detected: there is no bearing to steer by.
        # yaw_on must decide what to do with that, and by default it coasts.
        self.bearing_error_deg = bearing


def track_operator(lm: Optional[Landmarks], gestures_cfg: dict,
                   hfov_deg: float) -> OperatorTrack:
    """Detection state for one camera frame. Thresholds are the classifier's own."""
    if lm is None:
        return OperatorTrack(False, 0.0, NO_LANDMARKS)
    torso = torso_height(lm)
    if not _all_visible(lm, gestures_cfg["min_visibility"]):
        return OperatorTrack(False, torso, LOW_VISIBILITY)
    if torso < gestures_cfg["min_torso_height"]:
        return OperatorTrack(False, torso, SMALL_TORSO)
    return OperatorTrack(True, torso, DETECTED,
                         bearing_error_deg(_shoulder_mid_x(lm), hfov_deg))


LOSS_REASONS = (NO_LANDMARKS, LOW_VISIBILITY, SMALL_TORSO)


class TrackingLoss:
    """Accumulates the mission's primary metric: the det=N rate, and WHY."""

    def __init__(self):
        self.frames = 0
        self.lost = 0
        self.reasons = {r: 0 for r in LOSS_REASONS}

    def update(self, track: OperatorTrack) -> None:
        self.frames += 1
        if not track.detected:
            self.lost += 1
            self.reasons[track.loss_reason] += 1

    @property
    def fraction(self) -> Optional[float]:
        """Fraction of mission camera frames with no operator detected.

        None when the mission produced no camera frames -- an empty run has no tracking
        loss to report, and 0.0 would be a fabrication.
        """
        return None if not self.frames else self.lost / self.frames

    def reason_share(self, reason: str) -> Optional[float]:
        """That reason's share OF ALL FRAMES (not of lost frames), so the three shares
        sum to tracking_loss_fraction and can be compared across runs directly."""
        return None if not self.frames else self.reasons[reason] / self.frames
