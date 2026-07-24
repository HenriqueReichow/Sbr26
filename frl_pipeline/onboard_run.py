"""Entry point: onboard-perception gesture control of uav0 in BiguaSim.

Structure follows the working test.py: build a scenario dict, biguasim.make(...),
then a step loop that reads state["RGBCamera"] and shows it with cv2 (quit on 'q').
The only additions are: run the frame through onboard perception, classify a
gesture, and send the mapped cmd_vel instead of a constant [0, 0, 0].

    # full pipeline, real perception, real sim
    python -m frl_pipeline.onboard_run --config config.toml \
        --run-id op7_full_r1 --operator-id op7 --mode full --lighting normal

    # baseline ablation (no debounce), same code path otherwise
    python -m frl_pipeline.onboard_run --mode baseline_debounce_off

    # offline: replayed landmarks, no ZED / mediapipe / simulator
    python -m frl_pipeline.onboard_run --perception replay --dry-run

The ZED streams the operator's body into the world via LiveLink (engine-side); this
process never touches the ZED SDK. Gestures are recovered from the drone's own
camera, which is the point of the architecture.
"""
from __future__ import annotations

import argparse
import time

from .config import load_config
from .gesture_control import GestureController
from .gesture_logging import GestureLogger
from .gestures import classify
from .perception import make_pose_estimator


def build_scenario(cfg: dict) -> dict:
    """Scenario dict mirroring test.py: cmd_vel, DynamicsSensor pair, RGBCamera."""
    ob = cfg["onboard"]
    pose_key = cfg["world"]["pose"]["sensor_key"]
    return {
        "package_name": ob["package_name"],
        "world": ob["world"],
        "main_agent": ob["main_agent"],
        "agents": [
            {
                "agent_name": ob["main_agent"],
                "agent_type": ob["agent_type"],
                "sensors": [
                    {
                        "sensor_type": "DynamicsSensor",
                        "socket": "IMUSocket",
                        "configuration": {"UseCOM": True, "UseRPY": False},
                    },
                    {
                        "sensor_type": "DynamicsSensor",
                        "sensor_name": pose_key,
                        "socket": "IMUSocket",
                        "configuration": {"UseCOM": False, "UseRPY": True},
                    },
                    {
                        "sensor_type": "RGBCamera",
                        "sensor_name": "RGBCamera",
                        "socket": "CameraSocket",
                        "Hz": ob["camera_hz"],
                        "configuration": {
                            "CaptureWidth": ob["capture_width"],
                            "CaptureHeight": ob["capture_height"],
                        },
                    },
                ],
                "dynamics": {"batch_size": 1},
                "control_abstraction": ob["control_abstraction"],
                "location": ob["location"],
                "rotation": ob["rotation"],
            }
        ],
    }


def read_pose(state: dict, pose_cfg: dict):
    """Ground-truth pose ((x,y,z),(roll,pitch,yaw)) from the DynamicsSensor vector.

    Index layout VERIFIED on this build: an 18-float vector, position at [6:9] (a drone
    spawned at [-2.2, 0, 1.0] reads exactly that), rpy in DEGREES at [15:18] (rotation
    [0,0,90] reads yaw = 90.0). test2.py's [0:2] read the wrong field.
    """
    vec = state.get(pose_cfg["sensor_key"])
    if vec is None:
        return None
    pi, ri = pose_cfg["position_indices"], pose_cfg["rpy_indices"]
    return (
        (float(vec[pi[0]]), float(vec[pi[1]]), float(vec[pi[2]])),
        (float(vec[ri[0]]), float(vec[ri[1]]), float(vec[ri[2]])),
    )


def read_velocity(state: dict, pose_cfg: dict):
    """Ground-truth WORLD velocity (vx, vy, vz) from the DynamicsSensor vector.

    base_model.py:66 reads these same indices as the vehicle's velocity, and they track
    the commanded axis under `accel`. No finite-differencing needed.
    """
    vec = state.get(pose_cfg["sensor_key"])
    if vec is None:
        return None
    vi = pose_cfg["velocity_indices"]
    return (float(vec[vi[0]]), float(vec[vi[1]]), float(vec[vi[2]]))


def parse_args():
    p = argparse.ArgumentParser(description="Onboard gesture control for uav0")
    p.add_argument("--config", default="config.toml")
    p.add_argument("--run-id", dest="run_id", default=None)
    p.add_argument("--operator-id", dest="operator_id", default=None)
    p.add_argument("--mode", choices=["full", "baseline_debounce_off"], default=None)
    p.add_argument("--lighting", default=None)
    p.add_argument("--perception", choices=["mediapipe", "replay"], default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="skip biguasim entirely; drive from replayed landmarks only")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    for key in ("run_id", "operator_id", "mode", "lighting"):
        if getattr(args, key) is not None:
            cfg["run"][key] = getattr(args, key)
    if args.perception is not None:
        cfg["perception"]["backend"] = args.perception

    estimator = make_pose_estimator(cfg)
    logger = GestureLogger(cfg["run"], cfg["logging"]["out_dir"])
    controller = GestureController(cfg, logger, time.time)

    env = cv2 = None
    if not args.dry_run:
        import biguasim
        import cv2 as _cv2
        cv2 = _cv2
        env = biguasim.make(scenario_cfg=build_scenario(cfg), verbose=True)

    pose_cfg = cfg["world"]["pose"]
    command = list(cfg["gesture_command_map"]["none"])
    frame_id = 0
    printed_raw = False

    try:
        for step_i in range(cfg["onboard"]["max_ticks"]):
            if env is None:
                state, pose, image = {}, None, None
            else:
                state = env.step(list(command))[cfg["onboard"]["main_agent"]][0]
                if not printed_raw and pose_cfg["sensor_key"] in state:
                    print("[POSE-CALIB] raw", list(state[pose_cfg["sensor_key"]]))
                    printed_raw = True
                pose = read_pose(state, pose_cfg)
                if "RGBCamera" not in state:
                    continue  # camera runs slower than the physics tick
                image = state["RGBCamera"][:, :, 0:3].astype("uint8")

            # Recognition inference time: pose estimation + command inference only.
            t_infer = time.perf_counter()
            landmarks = estimator.landmarks(image)
            raw = classify(landmarks, cfg["gestures"], set(cfg["gestures"]["enabled"]))
            inference_s = time.perf_counter() - t_infer

            command = controller.update(raw, pose, frame_id, inference_s)
            frame_id += 1

            if image is not None:
                estimator.annotate(image)
                cv2.putText(image, controller.active_gesture, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("Camera Output", image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.max_steps is not None and frame_id >= args.max_steps:
                break
            if env is None and getattr(estimator, "exhausted", lambda: False)():
                break
    finally:
        estimator.close()
        logger.close()
        if cv2 is not None:
            cv2.destroyAllWindows()
    print(f"[done] {frame_id} camera frames; logs in {logger.run_dir}")


if __name__ == "__main__":
    main()
