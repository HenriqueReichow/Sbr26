# Onboard Pose-Command Pipeline for a BiguaSim Drone

Control a simulated drone with your body. The drone's **own onboard camera** observes a
human operator; [MediaPipe](https://developers.google.com/mediapipe) recovers the
operator's 2D body pose from that image; a small rule-based classifier maps the pose to
one interaction command; and the drone executes it. Recognition runs **onboard the
vehicle** — the drone acts only on what it can see.

```
drone RGB frame ─▶ MediaPipe landmarks ─▶ gesture classifier ─▶ velocity command ─▶ drone
```

## Command vocabulary

| Command     | Pose                                                   | Action          |
|-------------|--------------------------------------------------------|-----------------|
| `go_left`   | left arm out to the side, right arm down               | translate left  |
| `go_right`  | right arm out to the side, left arm down               | translate right |
| `ascend`    | both arms straight overhead                            | climb           |
| `land`      | left arm overhead, right arm out to the side           | descend         |
| `yaw_right` | both arms out to the sides, elbows bent, forearms up   | rotate          |

The exact angle bands and the command→velocity mapping live in `config.toml`
(`[gestures]`, `[gesture_command_map]`).

## Repository layout

```
frl_pipeline/
  perception.py       drone RGB frame -> 2D body landmarks (MediaPipe)
  gestures.py         landmarks -> one command (shoulder/elbow angles + debounce)
  tracking.py         operator bearing from the image (keep them in frame)
  gesture_control.py  command -> velocity setpoint
  gesture_logging.py  per-frame CSV of the recognised command + timing
  flight.py           velocity/attitude control under the `accel` abstraction
  zed_source.py       read frames from a real ZED camera
  onboard_run.py      entry point: build the scenario, run the perception+command loop
config.toml           all thresholds, gains, camera and scenario settings
```

---

## Installation

### 1. Python environment

Python **3.11** is recommended (MediaPipe wheels are published for it).

```bash
python3.11 -m venv .venv && source .venv/bin/activate   # or a conda env
pip install -r requirements.txt
```

### 2. Install BiguaSim

BiguaSim is the drone simulator. It is **not** on PyPI — install the Python client from
its repository (see the official
[Installation guide](https://github.com/hydrone-furg/biguasim)).

```bash
git clone https://github.com/hydrone-furg/biguasim.git biguasim
cd biguasim
pip install .
cd ..
```

The Python client is lightweight; the world **binaries** are placed separately (next
step).

### 3. Install the Competition world

This pipeline runs in the **CompetitionMap** world (package `Competition`). It is a large
build and is distributed out-of-band, not via git or the BiguaSim package server.

1. Download it from Google Drive: **[⟶ download the Competition world](TODO-DRIVE-LINK)**.
2. Unpack it into your BiguaSim worlds directory. On Linux, packages live at
   `~/.local/share/biguasim/<biguasim_version>/worlds/`, and each package is one folder
   inside it. This project uses BiguaSim **1.0.0**, so the world must end up at:

   ```
   ~/.local/share/biguasim/1.0.0/worlds/Competition
   ```

   That folder must contain `config.json` and the `Linux/` build directory — i.e.
   `~/.local/share/biguasim/1.0.0/worlds/Competition/config.json` must exist.

> If your installed BiguaSim reports a different version, place the world under that
> version's `worlds/` folder instead (the path is version-partitioned). You can also
> override the location entirely with the `HOLODECKPATH` environment variable.

Verify it is found:

```python
from biguasim import packagemanager
print(packagemanager.get_binary_path_for_package("Competition"))
```

### 4. ZED camera (the operator source)

The operator is a **real person seen by a ZED stereo camera**. The ZED streams the
operator's body into the simulator over **LiveLink** (engine-side, via the Stereolabs ZED
SDK), where BiguaSim renders it as an avatar in front of the drone. This Python process
never talks to the ZED SDK directly — it only reads the drone's rendered camera.

- Install the [ZED SDK](https://www.stereolabs.com/developers) and its Python API:
  ```bash
  python /usr/local/zed/get_python_api.py
  ```
- Enable the ZED's LiveLink body-tracking stream so the simulator receives the operator.

`frl_pipeline/zed_source.py` is provided for reading raw ZED RGB frames directly, if you
need it.

---

## Run

With the world installed and the operator streaming in:

```bash
python -m frl_pipeline.onboard_run --config config.toml
```

A window shows the drone's onboard camera with the detected skeleton and the currently
recognised command. Stand at the operator position, hold a pose from the table above, and
the drone obeys. Press `q` to quit.

Per-frame recognition logs are written under `logs/` (configurable in `config.toml`,
`[logging]`).
