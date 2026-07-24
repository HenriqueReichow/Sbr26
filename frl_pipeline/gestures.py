"""Interaction command recognition from 2D onboard-camera landmarks (paper Table I).

Recognition runs on what the SIMULATED DRONE SEES. The ZED only streams the
operator's body into the world (LiveLink, engine-side); the drone's camera observes
the rendered avatar and this module maps the observed body pose to one command.

Vocabulary (paper Table I), five commands:

    go_left     left arm horizontal to side, right arm down
    go_right    right arm horizontal to side, left arm down
    ascend      both arms overhead
    land        left arm overhead, right arm horizontal to the side
    yaw_right   both arms out to the sides, elbows BENT, forearms up (double-biceps)

Following paper Sec. IV-E, each arm contributes a shoulder elevation angle theta
(upper-arm vs the downward body axis) and an elbow flexion angle phi (upper-arm vs
forearm), both in degrees. Here they are measured in the IMAGE PLANE rather than on
3D joints, because the drone's camera is the sensor.

Each arm's theta falls in one of three disjoint bands -- `down`, `side`, `up` -- and
the four commands are distinct pairs of bands:

    (side, down) go_left      (down, side) go_right
    (up,   up)   ascend       (up,   side) land

Every command is a distinct band pair, and an arm cannot be in two bands at once, so
no two commands can collide. The bands are non-adjacent by construction -- angles
between them fall in a dead zone and yield no command -- so a pose stays unambiguous
while an arm is still swinging.

`yaw_right` is the double-biceps: both arms out to the sides (side band) with the elbows
BENT and forearms up. It is the only pose in the side band with bent elbows -- the side
commands (go_left, go_right, land) all require a STRAIGHT side arm (phi >= extended_min),
so a bent side arm is otherwise rejected. Both arms bent-and-side is a distinct pair that
collides with nothing. (The T-pose (side, side) with EXTENDED arms maps to no command; it
was the bilaterally-ambiguous pose the original `land` failed on.)

`land` is deliberately ASYMMETRIC. It was a T-pose (side, side), and MediaPipe's
tracker flip-flopped the two arms of the bilaterally symmetric, untextured avatar: once
perturbed by the arms sweeping up it never re-locked, and theta/phi on one arm wandered
across bands mid-hold. `ascend` (up, up) is symmetric too but survives, because a
left/right swap maps it onto itself. Only poses that differ per arm can be corrupted by
a swap, so those must not be mirror-ambiguous.

Both `side` arms must additionally be extended (phi >= extended_min), so a bent arm
resting near horizontal does not read as a command.

All angle thresholds are config. `torso_height` (shoulder-to-hip extent in the image)
normalises the distance gate, so it holds as the drone changes its standoff.

Image convention: x,y normalised to [0,1]; y grows DOWNWARD.
"""
from __future__ import annotations

import math
from typing import Optional, Set, Tuple

from .perception import Landmarks

NONE = "none"
COMMANDS = ("go_left", "go_right", "ascend", "land", "yaw_right")

Vec2 = Tuple[float, float]


# --- 2D helpers (kept local; the pipeline stays numpy-free) -------------------

def _sub(a: Vec2, b: Vec2) -> Vec2:
    return (a[0] - b[0], a[1] - b[1])


def _norm(a: Vec2) -> float:
    return math.hypot(a[0], a[1])


def angle_deg(a: Vec2, b: Vec2) -> float:
    """Angle between two image-plane vectors, degrees. 0.0 if either is degenerate."""
    na, nb = _norm(a), _norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    c = (a[0] * b[0] + a[1] * b[1]) / (na * nb)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _xy(lm: Landmarks, name: str) -> Vec2:
    p = lm[name]
    return (p[0], p[1])


def _mid(lm: Landmarks, a: str, b: str) -> Vec2:
    pa, pb = _xy(lm, a), _xy(lm, b)
    return (0.5 * (pa[0] + pb[0]), 0.5 * (pa[1] + pb[1]))


def torso_height(lm: Landmarks) -> float:
    """Shoulder-to-hip vertical extent: the scale the distance gate is expressed in."""
    return abs(_mid(lm, "left_hip", "right_hip")[1]
               - _mid(lm, "left_shoulder", "right_shoulder")[1])


# --- Per-arm features ---------------------------------------------------------

def arm_features(lm: Landmarks, side: str) -> Tuple[float, float]:
    """(theta, phi) for one arm, per paper Sec. IV-E, measured in-image.

    theta  upper-arm vs the downward body axis (shoulder -> hip)
    phi    upper-arm vs forearm, at the elbow (180 = straight)
    """
    shoulder = _xy(lm, f"{side}_shoulder")
    elbow = _xy(lm, f"{side}_elbow")
    wrist = _xy(lm, f"{side}_wrist")
    hip = _xy(lm, f"{side}_hip")

    theta = angle_deg(_sub(elbow, shoulder), _sub(hip, shoulder))
    phi = angle_deg(_sub(shoulder, elbow), _sub(wrist, elbow))
    return theta, phi


def _theta_band(theta: float, thr: dict) -> str:
    """One of 'down' | 'side' | 'up', or 'mid' for the dead zones between them."""
    if theta <= thr["down_max"]:
        return "down"
    if thr["side_min"] <= theta <= thr["side_max"]:
        return "side"
    if theta >= thr["up_min"]:
        return "up"
    return "mid"


def _all_visible(lm: Landmarks, min_visibility: float) -> bool:
    return all(v >= min_visibility for (_, _, v) in lm.values())


# (left band, right band) -> command. Disjoint by construction; see module docstring.
_BAND_PAIRS = {
    ("side", "down"): "go_left",
    ("down", "side"): "go_right",
    ("up", "up"): "ascend",
    ("up", "side"): "land",
}


def classify(lm: Optional[Landmarks], cfg: dict, enabled: Set[str]) -> str:
    """Return one of COMMANDS, or NONE if no band pair matches this frame."""
    if lm is None or not _all_visible(lm, cfg["min_visibility"]):
        return NONE

    if torso_height(lm) < cfg["min_torso_height"]:
        # Operator too far / too foreshortened for the image angles to mean anything.
        return NONE

    t_thr, p_thr = cfg["theta"], cfg["phi"]
    th_l, ph_l = arm_features(lm, "left")
    th_r, ph_r = arm_features(lm, "right")
    band_l, band_r = _theta_band(th_l, t_thr), _theta_band(th_r, t_thr)

    # Double-biceps: both arms out to the side (side band) with elbows BENT (forearms up).
    # This is the yaw_right pose -- a natural "rotate" gesture, and the ONE pose in the
    # side band with bent elbows (the side commands go_left/go_right/land all require a
    # STRAIGHT side arm, phi >= extended_min). (side, side) with EXTENDED arms is the old
    # bilaterally-ambiguous T-pose and maps to no command.
    if band_l == "side" and band_r == "side":
        ext = p_thr["extended_min"]
        if ph_l < ext and ph_r < ext:
            return "yaw_right" if "yaw_right" in enabled else NONE
        return NONE

    candidate = _BAND_PAIRS.get((band_l, band_r), NONE)
    if candidate == NONE:
        return NONE

    # A horizontal arm only counts if it is actually extended, not merely bent so that
    # the elbow happens to sit near shoulder height.
    if band_l == "side" and ph_l < p_thr["extended_min"]:
        return NONE
    if band_r == "side" and ph_r < p_thr["extended_min"]:
        return NONE

    return candidate if candidate in enabled else NONE


class Debouncer:
    """Emit a command only after it has held for H consecutive frames (paper IV-E).

    H == 1 degenerates to "emit on the first matching frame", which is exactly the
    `debounce_off` baseline ablation the paper reports against (Sec. V-B).
    """

    def __init__(self, hold_frames: int):
        self.H = max(1, int(hold_frames))
        self._raw = NONE
        self._count = 0
        self._stable = NONE

    def update(self, raw: str) -> str:
        if raw == self._raw:
            self._count += 1
        else:
            self._raw = raw
            self._count = 1
        if self._count >= self.H:
            self._stable = raw
        return self._stable
