"""Flight control for the `accel` abstraction: hold framing, fly the mission phases.

`accel` is a misnomer, and this module is written against what the simulator MEASURABLY
does rather than what the name suggests (see tools/characterize_plant.py):

  * The command is a first-order-lagged VELOCITY setpoint in the WORLD frame:
    v(t) = A(1 - e^(-t/tau)), settling at exactly A, with tau ~= 1 s.
  * A zero command therefore DECAYS velocity to zero. It does not coast.
  * There is a hard DEAD ZONE: a command whose vector NORM is <= ~0.355 moves the drone
    not at all. Above it, each axis tracks its own component, so direction is preserved.
  * Stopping distance from speed v after commanding zero is exactly v * tau.

Two consequences drive every design choice below.

First, a P-controller cannot settle. Inside min_command/k of its setpoint it commands a
velocity the engine discards, so it stalls with that much error still standing. With the
old gains that floor was 0.71 m -- wider than landing_radius, which is why no landing
was ever scored. So the position loops here are BANG-BANG: drive at cruise_speed, and
cut to zero exactly one coast-length (v * tau) short of the target. The first-order
decay then lands the drone on the mark. That is a deadbeat approach, and it is only
possible because the coast length is an exact, measured function of speed.

Second, there is no point closing a velocity loop in Python around a plant that is
already a velocity servo. The desired world velocity IS the command.

Velocity is read straight from the DynamicsSensor's world-velocity field; the previous
finite-difference of position was unnecessary.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

Vec3 = Tuple[float, float, float]

# --- Experimental conditions: how much of the recognized command the drone executes --
#
# The perception path is IDENTICAL in all three -- same camera, same estimator, same
# classifier, same fixed command sequence. Only the drone's self-motion differs, which
# is the independent variable: how much does the drone's own movement degrade its own
# onboard recognition?
STATIC = 0        # hold station: recognize and log, execute nothing
ALTITUDE = 1      # execute the vertical axis only (ascend / land)
FULL = 2          # execute everything (orbit / strafe as well)

CONDITIONS = (STATIC, ALTITUDE, FULL)
CONDITION_NAMES = {STATIC: "static", ALTITUDE: "altitude-only", FULL: "full-closed-loop"}

# --- Mission phases -----------------------------------------------------------
# The gesture phase is bounded by the CAMERA-FRAME clock, not wall time, so the command
# sequence is bit-identical across conditions. Landing is a separate scripted phase: the
# gesture vocabulary has no "fly to a base" command, and the standoff hold makes the
# bases geometrically unreachable while it is engaged.
#
# APPROACH sits between TAKEOFF and GESTURE: a spawn need not be anywhere near the
# operator (see [onboard].location), and yaw_on's bearing signal only exists once the
# operator is already in frame -- it cannot bootstrap itself from an arbitrary spawn
# heading. APPROACH is scripted navigation on ground truth, exactly like LAND already is
# (see FlightController._approach_velocity / _land_velocity); it is a transit, not a
# claim about perception. GESTURE only begins once the drone is already within the
# standoff band, so the vision-driven experiment always starts from a known-good pose.
TAKEOFF = "takeoff"
APPROACH = "approach"
GESTURE = "gesture"
LAND = "land"
DONE = "done"
PHASES = (TAKEOFF, APPROACH, GESTURE, LAND, DONE)

# --- The yaw study's independent variable ------------------------------------
# The camera cannot pan or tilt, so framing is the airframe's job. yaw_on closes that
# loop on the drone's own camera; yaw_off freezes the heading. The standoff hold tracks
# the operator in BOTH, so heading is the only difference.
YAW_ON = "yaw_on"
YAW_OFF = "yaw_off"
YAW_MODES = (YAW_ON, YAW_OFF)


def gate_gesture(gesture_body: Vec3, condition: int) -> list:
    """Mask the recognized command's velocity down to what this condition executes."""
    vx, vy, vz = gesture_body
    if condition == STATIC:
        return [0.0, 0.0, 0.0]
    if condition == ALTITUDE:
        return [0.0, 0.0, vz]
    if condition == FULL:
        return [vx, vy, vz]
    raise ValueError(f"unknown motion condition: {condition!r} (expected one of {CONDITIONS})")


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def body_to_world(cmd: Vec3, yaw_deg: float) -> list:
    """Rotate a body-frame [vx, vy, vz] into the world frame by the drone's yaw."""
    vx, vy, vz = cmd
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return [vx * c - vy * s, vx * s + vy * c, vz]


def _wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def _norm(v) -> float:
    return math.sqrt(sum(c * c for c in v))


class FlightController:
    """Turns a mission phase and a body-frame gesture velocity into an `accel` command."""

    def __init__(self, cfg: dict, dt: float):
        self.ob = cfg["onboard"]
        self.mi = cfg["mission"]
        self.pl = cfg["plant"]
        self.land_z_max = cfg["world"]["platforms"]["land_z_max"]
        self.operator_xy = cfg["human"]["livelink"]["root_position"][:2]
        self.dt = dt

        self.tau = float(self.pl["tau_s"])
        self.min_command = float(self.pl["min_command"])
        self.stop_speed = float(self.pl["stop_speed"])
        self.cruise = float(self.pl["cruise_speed"])
        self.arrive_tol = float(self.pl["arrive_tol_m"])

        yaw = cfg["yaw"]
        self.yaw_mode = yaw["mode"]
        if self.yaw_mode not in YAW_MODES:
            raise ValueError(f"[yaw].mode = {self.yaw_mode!r} (expected one of {YAW_MODES})")
        self.fixed_heading_deg = float(yaw["fixed_heading_deg"])
        self.loss_recovery = bool(yaw["loss_recovery"])
        self.search_rate_deg_s = float(yaw["search_rate_deg_s"])
        self._last_seen_bearing_deg: Optional[float] = None

        # Angular position to hold around the operator during GESTURE (world degrees,
        # bearing from operator to drone). None until dual_goal_mission sets it at the
        # APPROACH->GESTURE handoff -- see _gesture_velocity.
        self.hold_bearing_deg: Optional[float] = None

        self._prev_yaw: Optional[float] = None
        # Latched once the land phase has stopped translating, so the terminal coast is
        # never re-armed by the drone drifting a centimetre back out of tolerance.
        self._xy_parked = False

    # --- the deadbeat primitive ------------------------------------------------
    def _approach(self, err: float, err_rate: float, tol: float, speed: float) -> float:
        """Desired rate along one coordinate, driving `err` -> 0 without overshoot.

        `err` is (target - current); `err_rate` is d(err)/dt. Returns a rate with the
        sign of `err`, or 0.0 when the drone is either close enough or already carrying
        exactly enough momentum to coast the rest of the way in.
        """
        if abs(err) <= tol:
            return 0.0
        # Closing (err and err_rate have opposite signs) and one coast-length away: cut
        # the command and let the first-order decay do the last v * tau metres.
        if err * err_rate < 0.0 and abs(err) <= abs(err_rate) * self.tau:
            return 0.0
        return math.copysign(speed, err)

    def _deadband(self, v_world) -> list:
        """Discard commands the engine would ignore; scale the rest clear of the floor.

        Direction is preserved because the dead zone is on the vector norm, not per-axis
        (measured). Below `stop_speed` the intent is "hold", so command exactly zero.
        """
        n = _norm(v_world)
        if n < self.stop_speed:
            return [0.0, 0.0, 0.0]
        if n < self.min_command:
            k = self.min_command / n
            return [c * k for c in v_world]
        return list(v_world)

    # --- per-phase desired world velocity --------------------------------------
    def _radial(self, pos: Vec3, vel: Vec3):
        """(distance to operator, d(distance)/dt, unit vector pointing AWAY from them)."""
        dx, dy = pos[0] - self.operator_xy[0], pos[1] - self.operator_xy[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return 0.0, 0.0, (1.0, 0.0)
        return dist, (vel[0] * dx + vel[1] * dy) / dist, (dx / dist, dy / dist)

    def _vertical(self, pos: Vec3, vel: Vec3, target_z: float, speed: float,
                  tol: float) -> float:
        return self._approach(target_z - pos[2], -vel[2], tol, speed)

    def _takeoff_velocity(self, pos: Vec3, vel: Vec3) -> list:
        """Climb to hold_altitude. Horizontal is left to decay to a stop."""
        return [0.0, 0.0,
                self._vertical(pos, vel, self.ob["hold_altitude"], self.cruise,
                               self.arrive_tol)]

    def _approach_velocity(self, pos: Vec3, vel: Vec3) -> list:
        """Fly straight at the operator (ground truth XY) to get within camera range
        before GESTURE begins. A transit, like _land_velocity -- not the experiment."""
        dist, _, radial_out = self._radial(pos, vel)
        if dist < 1e-6:
            vx, vy = 0.0, 0.0
        else:
            vx, vy = -radial_out[0] * self.cruise, -radial_out[1] * self.cruise
        vz = self._vertical(pos, vel, self.ob["hold_altitude"], self.cruise, self.arrive_tol)
        return [vx, vy, vz]

    def _gesture_velocity(self, pos: Vec3, vel: Vec3, gesture_body: Vec3,
                          yaw_deg: float) -> list:
        """Gesture velocity superimposed on the standoff + altitude hold, in world frame.

        `ascend` and `land` own the vertical axis while they are active -- moving in z is
        what they mean -- so the altitude hold yields to them.

        The standoff hold is computed in the WORLD frame, along the operator->drone
        radial. It must NOT be a body-frame "forward" velocity: body +x points at the
        operator only while the drone is facing them, which is precisely what yaw_off does
        not do. Expressed body-frame, a frozen-heading drone that strafes sideways pushes
        itself along a fixed world axis and runs away to infinity (measured: 13.9 m out
        and climbing). Keeping the hold radial is what makes yaw the ONLY difference
        between the two conditions.

        The gesture velocity stays body-frame: `go_left` means the drone's left.
        """
        ob = self.ob
        vx_g, vy_g, vz_g = gesture_body

        dist, dist_rate, radial_out = self._radial(pos, vel)
        # Controlled coordinate is `dist`, target `standoff_m`. A positive desired rate
        # means "let the distance grow", i.e. move along the outward radial.
        radial_rate = self._approach(ob["standoff_m"] - dist, -dist_rate,
                                     ob["standoff_tol_m"], self.cruise)

        # The radial term above only controls DISTANCE -- nothing constrains the ANGULAR
        # position around the operator, so residual tangential velocity at the GESTURE
        # handoff (left over from APPROACH) is never damped and the drone coasts to
        # whatever angle it happens to stop at, which can be well outside the camera's
        # cone even though distance is held perfectly (measured: a stationary operator
        # still saw the drone drift ~2m sideways and settle at a wrong bearing, killing
        # detection for the rest of the phase). Add a tangential P-term holding the
        # bearing (drone as seen from the operator) at whatever it was when GESTURE
        # started, exactly like fixed_heading_deg freezes yaw at that same moment.
        tangential_rate = 0.0
        tangent_out = (0.0, 0.0)
        if self.hold_bearing_deg is not None and dist > 1e-6:
            # radial_out points operator->drone; rotate +90deg for the tangential axis.
            tangent_out = (-radial_out[1], radial_out[0])
            # radial_out is operator->drone, matching hold_bearing_deg's own convention
            # (set in dual_goal_mission.py from the same pos-operator_xy difference).
            current_bearing = math.degrees(math.atan2(radial_out[1], radial_out[0]))
            bearing_err_deg = _wrap_deg(self.hold_bearing_deg - current_bearing)
            # Small-angle: arc length to correct = dist * angle(rad). Reuse _approach so
            # the same standoff_tol_m-scale deadband/cruise cap applies as the radial term.
            tangential_rate = self._approach(
                dist * math.radians(bearing_err_deg), 0.0, ob["standoff_tol_m"], self.cruise)

        if vz_g > 0.0 and pos[2] >= ob["max_altitude"]:
            vz_g = 0.0                  # `ascend` stops before the operator leaves frame
        vz = vz_g if vz_g != 0.0 else self._vertical(pos, vel, ob["hold_altitude"],
                                                     self.cruise, self.arrive_tol)

        gesture_world = body_to_world((vx_g, vy_g, 0.0), yaw_deg)
        return [radial_rate * radial_out[0] + tangential_rate * tangent_out[0]
                + gesture_world[0],
                radial_rate * radial_out[1] + tangential_rate * tangent_out[1]
                + gesture_world[1],
                vz]

    def _land_velocity(self, pos: Vec3, vel: Vec3, base_pos) -> list:
        """Fly to the base in xy at cruising altitude, then descend. Standoff released."""
        dx, dy = base_pos[0] - pos[0], base_pos[1] - pos[1]
        dist = math.hypot(dx, dy)
        v_xy = math.hypot(vel[0], vel[1])

        if not self._xy_parked:
            closing = (vel[0] * dx + vel[1] * dy) > 0.0
            if dist <= self.arrive_tol or (closing and dist <= v_xy * self.tau):
                self._xy_parked = True
            else:
                # Hold altitude on the way across; descending early risks clipping the
                # rendered operator, who stands between the drone and base1.
                vz = self._vertical(pos, vel, self.ob["hold_altitude"], self.cruise,
                                    self.arrive_tol)
                return [self.cruise * dx / dist, self.cruise * dy / dist, vz]

        # Descend toward touchdown_z, which sits below the floor on purpose: the floor,
        # not the setpoint, arrests the drone. See landed().
        return [0.0, 0.0,
                self._vertical(pos, vel, self.mi["touchdown_z"],
                               self.mi["descent_speed"], self.mi["z_arrive_tol_m"])]

    def _desired_yaw_rate(self, yaw_deg: float, bearing_error_deg) -> float:
        """rad/s the airframe should be turning at, per the yaw condition.

        yaw_off holds a fixed heading. yaw_on steers on the bearing the tracking module
        read out of the drone's OWN camera -- never ground truth. When the operator is
        not detected there is no bearing, and the default is to command zero: no signal,
        no rotation. That makes tracking loss absorbing, which is the honest behaviour of
        a vehicle whose only sensor is a fixed camera.
        """
        ob = self.ob
        if self.yaw_mode == YAW_OFF:
            err = _wrap_deg(self.fixed_heading_deg - yaw_deg)
            return ob["k_yaw"] * math.radians(err)

        if bearing_error_deg is not None:
            return ob["k_yaw"] * math.radians(bearing_error_deg)

        if self.loss_recovery and self._last_seen_bearing_deg is not None:
            # Sweep back toward the side the operator was last seen on. A separate
            # condition, off by default; never enable it and still call the run yaw_on.
            return math.copysign(math.radians(self.search_rate_deg_s),
                                 self._last_seen_bearing_deg)
        return 0.0

    @property
    def last_seen_bearing_deg(self):
        """Logged every frame so a recovery condition can be added without re-running."""
        return self._last_seen_bearing_deg

    def _track_yaw_rate(self, yaw_deg: float, yaw_rate_des: float) -> float:
        """Cascaded P: track a DESIRED yaw rate via measured yaw-rate feedback."""
        ob = self.ob
        if self._prev_yaw is None:      # first tick: no yaw-rate estimate yet
            self._prev_yaw = yaw_deg
            return 0.0
        yaw_rate = math.radians(_wrap_deg(yaw_deg - self._prev_yaw)) / self.dt
        self._prev_yaw = yaw_deg
        return _clamp(ob["k_yaw_rate"] * (yaw_rate_des - yaw_rate),
                      -ob["max_ang_accel"], ob["max_ang_accel"])

    def _yaw_command(self, yaw_deg: float, bearing_error_deg) -> float:
        """Vision-driven yaw: heading error (from the camera bearing) -> angular accel."""
        # Record the last bearing the camera actually delivered, before any early exit --
        # otherwise the first tick's bearing is dropped and a later loss_recovery
        # condition has nothing to sweep toward.
        if bearing_error_deg is not None:
            self._last_seen_bearing_deg = bearing_error_deg
        yaw_rate_des = self._desired_yaw_rate(yaw_deg, bearing_error_deg)
        return self._track_yaw_rate(yaw_deg, yaw_rate_des)

    def _approach_yaw_command(self, yaw_deg: float, pos: Vec3) -> float:
        """Scripted yaw during APPROACH: face the operator directly on ground truth.

        Does NOT touch last_seen_bearing_deg -- that field is documented as vision-only
        (it seeds the loss_recovery sweep), and this phase runs before the operator has
        ever been seen by the camera at all.
        """
        yaw_rate_des = self.ob["k_yaw"] * math.radians(self.heading_error_deg(pos, yaw_deg))
        return self._track_yaw_rate(yaw_deg, yaw_rate_des)

    def heading_error_deg(self, pos: Vec3, yaw_deg: float) -> float:
        """Signed yaw error that would point the drone at the operator."""
        bearing = math.degrees(math.atan2(self.operator_xy[1] - pos[1],
                                          self.operator_xy[0] - pos[0]))
        return _wrap_deg(bearing - yaw_deg)

    # --- public ----------------------------------------------------------------
    def landed(self, pos: Vec3, vel: Vec3, base_pos) -> bool:
        """True once the drone has come to rest inside the base's landing box.

        Touchdown is "at rest, low enough", NOT "reached touchdown_z". The DjiMatrice's
        body origin rests ~0.28 m above the floor on its landing gear, and the bases are
        drawn without collision geometry, so any specific target z below that is
        unreachable and the descent would run until it timed out. The drone descends
        until the floor stops it; the floor is what ends the mission.
        """
        return (self._xy_parked
                and pos[2] <= base_pos[2] + self.land_z_max
                and _norm(vel) < self.stop_speed)

    def reset_for_next_takeoff(self) -> None:
        """Clear the LAND-phase latch so this controller can fly a second TAKEOFF ->
        APPROACH -> GESTURE -> LAND cycle to a different base in the same mission."""
        self._xy_parked = False

    def step(self, pose, vel: Vec3, gesture_body: Vec3, phase: str, base_pos=None,
             bearing_error_deg=None) -> list:
        """One `accel` command: [vx, vy, vz, 0, 0, alpha_z], world frame.

        `bearing_error_deg` is the camera-derived bearing to the operator, or None when
        the operator was not detected on the most recent camera frame. It is the ONLY
        operator signal the yaw loop sees; ground truth never reaches it.
        """
        pos, (_, _, yaw_deg) = pose

        if phase == TAKEOFF:
            v_world = self._takeoff_velocity(pos, vel)
        elif phase == APPROACH:
            v_world = self._approach_velocity(pos, vel)
        elif phase == GESTURE:
            v_world = self._gesture_velocity(pos, vel, gesture_body, yaw_deg)
        elif phase == LAND:
            if base_pos is None:
                raise ValueError("land phase needs base_pos")
            v_world = self._land_velocity(pos, vel, base_pos)
        elif phase == DONE:
            v_world = [0.0, 0.0, 0.0]
        else:
            raise ValueError(f"unknown mission phase: {phase!r} (expected one of {PHASES})")

        # APPROACH is scripted navigation, like LAND -- it faces the operator on ground
        # truth (there is no camera bearing to steer on yet; that's the whole reason this
        # phase exists). Every other phase steers only on the vision-derived bearing.
        yaw_alpha = (self._approach_yaw_command(yaw_deg, pos) if phase == APPROACH
                    else self._yaw_command(yaw_deg, bearing_error_deg))
        return self._deadband(v_world) + [0.0, 0.0, yaw_alpha]
