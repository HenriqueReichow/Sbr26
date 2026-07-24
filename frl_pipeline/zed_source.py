"""The ZED as a plain RGB camera: left-eye BGR frames, nothing else.

This is the first module in the project that touches the ZED SDK. Everywhere else, the
ZED's only job was to stream a body into Unreal over LiveLink so the map could render an
avatar; recognition then ran on the *drone's* simulated camera and this process never
imported the SDK (see onboard_run.py's docstring).

Biomechanical capture is different: it measures a REAL operator, so it needs real frames.
It does not need depth, positional tracking, or body tracking -- MediaPipe recovers the
landmarks, exactly as it does from the drone's camera, so the angle convention and units
match the recognizer's.

Requires the `pyzed` Python bindings, which are NOT installed by the ZED SDK by default:

    python /usr/local/zed/get_python_api.py

and a ZED plugged in (or an .svo recording passed as `svo_path`).
"""
from __future__ import annotations

from typing import Optional


class ZedRgbSource:
    """Left-eye BGR frames from a live ZED, or from an .svo recording."""

    def __init__(self, cfg: dict, svo_path: Optional[str] = None):
        try:
            import pyzed.sl as sl
        except ImportError as exc:      # pragma: no cover - needs the SDK + hardware
            raise RuntimeError(
                "biomech capture needs the ZED Python API. The SDK ships without it; "
                "install with:  python /usr/local/zed/get_python_api.py\n"
                "Then plug in a ZED, or pass --svo <recording.svo>."
            ) from exc
        import cv2

        self._sl = sl
        self._cv2 = cv2
        self._cam = sl.Camera()
        self._image = sl.Mat()

        init = sl.InitParameters()
        # Depth is the SDK's expensive stage and nothing here uses it.
        init.depth_mode = sl.DEPTH_MODE.NONE
        init.camera_fps = int(cfg["fps"])
        init.camera_resolution = getattr(sl.RESOLUTION, cfg["resolution"])
        if svo_path:
            init.set_from_svo_file(svo_path)

        status = self._cam.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(
                f"could not open the ZED: {status}. Is a camera attached? "
                f"(no /dev/video* and no Stereolabs USB device means no.)"
            )
        self._runtime = sl.RuntimeParameters()

    def frame(self):
        """One left-eye BGR frame, or None when the source is exhausted."""
        if self._cam.grab(self._runtime) != self._sl.ERROR_CODE.SUCCESS:
            return None                 # end of SVO, or a dropped grab
        self._cam.retrieve_image(self._image, self._sl.VIEW.LEFT)
        # get_data() is BGRA; MediaPipePoseEstimator expects BGR, as from the drone cam.
        return self._image.get_data()[:, :, 0:3].copy()

    def close(self) -> None:
        self._cam.close()
