# OpenXR Workflow

This repository now supports an optional OpenXR-style stereo rendering workflow without replacing the existing offline `render.py` path.

## What It Does

- Keeps the current Gaussian rendering core unchanged.
- Adds a stereo camera adapter that consumes per-eye OpenXR view data:
  - pose position
  - pose orientation quaternion in `xyzw`
  - per-eye asymmetric FOV (`angle_left/right/up/down`)
  - per-eye image size
- Writes:
  - `left_eye/*.png`
  - `right_eye/*.png`
  - `side_by_side/*.png` when enabled
  - `xr_session_manifest.json`

## Important Boundary

This repo does **not** directly create an OpenXR runtime, swapchain, or headset session.

Instead, it adds the renderer-side contract that an OpenXR host can call:

- `openxr_replay`: load recorded OpenXR-like view payloads from `.json` or `.jsonl`
- `openxr_socket`: receive newline-delimited JSON frames over TCP

That means the recommended delivery architecture is:

1. Keep HorizonGS as the stereo renderer.
2. Add a very thin OpenXR host in Unity, UE, or native code.
3. Let that host call `xrLocateViews`, obtain per-eye pose/FOV, and forward the data to this renderer.
4. Hand the rendered stereo images back to the host for runtime submission if needed.

## Replay Command

```bash
python render.py \
  -m /path/to/model \
  --xr_mode openxr_replay \
  --xr_input config/xr/openxr_frame_example.json \
  --xr_config config/xr/openxr_replay_example.yaml \
  --xr_output_name openxr_debug \
  --xr_output_layout both \
  --xr_save_video
```

## Socket Command

```bash
python render.py \
  -m /path/to/model \
  --xr_mode openxr_socket \
  --xr_config config/xr/openxr_replay_example.yaml \
  --xr_output_name openxr_socket \
  --xr_socket_host 127.0.0.1 \
  --xr_socket_port 6110
```

Each socket line must be one JSON frame. Send `{"type":"eos"}` to stop the session.

## Frame Schema

Minimal frame payload:

```json
{
  "frame_id": 0,
  "timestamp_ns": 0,
  "views": {
    "left": {
      "pose": {
        "position": [-0.032, 1.62, 0.0],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]
      },
      "fov": {
        "angle_left": -0.83,
        "angle_right": 0.74,
        "angle_up": 0.80,
        "angle_down": -0.80
      },
      "image_rect": {
        "width": 1832,
        "height": 1920
      }
    },
    "right": {
      "pose": {
        "position": [0.032, 1.62, 0.0],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]
      },
      "fov": {
        "angle_left": -0.74,
        "angle_right": 0.83,
        "angle_up": 0.80,
        "angle_down": -0.80
      },
      "image_rect": {
        "width": 1832,
        "height": 1920
      }
    }
  }
}
```

## Validation Checklist

1. Run `openxr_replay` with a single frame and verify that both eye images are generated.
2. Run a short recorded sequence and check `side_by_side.mp4` for stereo continuity.
3. Confirm the left and right eyes differ mainly by horizontal baseline, not by unexpected roll or vertical drift.
4. Confirm the scene origin and headset origin are aligned by tuning `scene_from_tracking` in the YAML config.
5. Once a host program is available, switch from `openxr_replay` to `openxr_socket` before doing runtime integration.

## Fast Debug Path

If the replay output looks meaningless, do not start with SteamVR. First validate the bridge with a known training camera:

```bash
python3 tools/make_xr_replay_from_camera.py \
  --cameras_json outputs/horizongs/real/road_subset/fine/cameras.json \
  --camera_name TIMELAPSE_0340 \
  --output /tmp/xr_from_known_camera.json \
  --ipd 0.0
```

Then render it:

```bash
python render.py \
  -m outputs/horizongs/real/road_subset/fine \
  --xr_mode openxr_replay \
  --xr_input /tmp/xr_from_known_camera.json \
  --xr_config config/xr/openxr_replay_example.yaml \
  --xr_output_name openxr_known_camera
```

Expected result:

- both eyes should look like a normal render
- because `--ipd 0.0`, left and right images should be almost identical

If that works, the renderer bridge is likely correct and the issue is the external OpenXR pose / scene alignment.
