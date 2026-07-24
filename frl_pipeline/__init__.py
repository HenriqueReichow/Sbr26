"""
frl_pipeline — onboard gesture control of a BiguaSim drone for the FRL Phase 3
(Human-Swarm Interaction) task.

A ZED stereo camera streams a real operator's body into the simulator via LiveLink
(engine-side); the SIMULATED DRONE'S OWN camera then observes that body and this
package recovers the interaction command from it. Recognition therefore runs onboard
the vehicle, matching the FRL onboard-perception constraint.

    perception.py       drone RGB frame -> 2D body landmarks (MediaPipe | replay)
    gestures.py         landmarks -> one of the Table I commands (theta/phi + debounce)
    gesture_control.py  command -> cmd_vel [vx, vy, vz], go-home nav, landing scoring
    gesture_logging.py  per-frame CSV: command, timestamps, measured latency
    onboard_run.py      scenario setup + step loop + cv2 view (structure of test.py)

Only perception.py's MediaPipe backend and onboard_run.py's simulator path touch
third-party packages (mediapipe, cv2, numpy, biguasim). Everything else is pure
Python, so the recognition pipeline runs and is unit-tested with neither the ZED SDK
nor the simulator present -- use `--perception replay --dry-run`.
"""
