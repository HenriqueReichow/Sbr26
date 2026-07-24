"""Onboard perception: drone RGB camera image -> 2D body landmarks.

The gesture classification runs on what the SIMULATED DRONE SEES, not on the ZED
joint stream. The ZED only streams the operator's body into the world (LiveLink,
engine-side); the drone's camera then observes the rendered avatar and this module
recovers landmarks from that image. That keeps perception where the paper claims it
is: onboard the vehicle.

All heavy deps (mediapipe, cv2) are imported lazily inside MediaPipePoseEstimator,
so the classifier, controller, logger and tests run on bare Python.
"""
from __future__ import annotations

import abc
import json
from typing import Dict, Optional, Tuple

# Landmark = (x, y, visibility), x/y normalised to [0,1] in image space, y grows DOWN.
Landmark = Tuple[float, float, float]
Landmarks = Dict[str, Landmark]

# The only landmarks the gesture rules need. Names match MediaPipe's PoseLandmark
# enum (lower-cased), so the index lookup below is a straight getattr.
LANDMARK_NAMES = (
    "nose",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
)


class PoseEstimator(abc.ABC):
    """Turns one drone-camera frame into named 2D landmarks (or None if no person)."""

    @abc.abstractmethod
    def landmarks(self, frame_bgr) -> Optional[Landmarks]:
        ...

    def annotate(self, frame_bgr) -> None:
        """Optionally draw the detected skeleton onto frame_bgr, in place."""

    def close(self) -> None:
        pass


class MediaPipePoseEstimator(PoseEstimator):
    """Real onboard perception. Needs mediapipe + opencv on the sim machine."""

    def __init__(self, cfg: dict):
        try:
            import mediapipe as mp
        except ImportError as exc:  # pragma: no cover - depends on the sim machine
            raise RuntimeError(
                "MediaPipePoseEstimator needs 'mediapipe' (and 'opencv-python'). "
                "Install them, or set [perception].backend = 'replay' to run offline."
            ) from exc
        import cv2

        self._cv2 = cv2
        self._mp = mp
        self._solution = mp.solutions.pose
        self._pose = self._solution.Pose(
            min_detection_confidence=cfg["min_detection_confidence"],
            min_tracking_confidence=cfg["min_tracking_confidence"],
            model_complexity=cfg["model_complexity"],
        )
        enum = self._solution.PoseLandmark
        self._idx = {n: getattr(enum, n.upper()).value for n in LANDMARK_NAMES}
        self._last = None

    def landmarks(self, frame_bgr) -> Optional[Landmarks]:
        rgb = self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb)
        self._last = result
        if result.pose_landmarks is None:
            return None
        lm = result.pose_landmarks.landmark
        return {
            name: (lm[i].x, lm[i].y, lm[i].visibility) for name, i in self._idx.items()
        }

    def annotate(self, frame_bgr) -> None:
        if self._last is None or self._last.pose_landmarks is None:
            return
        self._mp.solutions.drawing_utils.draw_landmarks(
            frame_bgr, self._last.pose_landmarks, self._solution.POSE_CONNECTIONS
        )

    def close(self) -> None:
        self._pose.close()


class ReplayPoseEstimator(PoseEstimator):
    """Test fixture: replays landmarks from JSONL, ignoring the image entirely.

    One JSON value per line: either null (no person detected that frame) or
    {"nose": [x, y, visibility], "left_shoulder": [...], ...}.

    Lets the whole gesture pipeline be exercised with no ZED, no MediaPipe and no
    simulator. It is a fixture, not an experimental condition.
    """

    def __init__(self, cfg: dict):
        self._loop = bool(cfg.get("loop", False))
        with open(cfg["path"]) as f:
            self._frames = [json.loads(line) for line in f if line.strip()]
        self._i = 0

    def landmarks(self, frame_bgr=None) -> Optional[Landmarks]:
        if self._i >= len(self._frames):
            if not self._loop or not self._frames:
                return None
            self._i = 0
        rec = self._frames[self._i]
        self._i += 1
        if rec is None:
            return None
        return {name: tuple(v) for name, v in rec.items()}

    def exhausted(self) -> bool:
        return not self._loop and self._i >= len(self._frames)


def make_pose_estimator(cfg: dict) -> PoseEstimator:
    backend = cfg["perception"]["backend"]
    if backend == "mediapipe":
        return MediaPipePoseEstimator(cfg["perception"]["mediapipe"])
    if backend == "replay":
        return ReplayPoseEstimator(cfg["perception"]["replay"])
    raise ValueError(f"unknown perception backend: {backend!r}")
