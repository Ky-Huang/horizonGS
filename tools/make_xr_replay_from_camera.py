import argparse
import json
import math
import os

import numpy as np


OPENXR_CAMERA_FROM_RENDER_CAMERA = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


def _load_cameras(path):
    with open(path, "r") as f:
        cameras = json.load(f)
    if not isinstance(cameras, list):
        raise ValueError("cameras.json must contain a list.")
    return cameras


def _find_camera(cameras, name):
    target = os.path.splitext(os.path.basename(name))[0].lower()
    for camera in cameras:
        camera_name = os.path.splitext(os.path.basename(camera["img_name"]))[0].lower()
        if camera_name == target:
            return camera
    raise ValueError(f"Camera '{name}' not found in cameras.json.")


def _rotation_matrix_to_quaternion_xyzw(rotation):
    m = np.asarray(rotation, dtype=np.float32)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s

    quat = np.asarray([x, y, z, w], dtype=np.float32)
    quat /= max(np.linalg.norm(quat), 1e-8)
    return quat


def _camera_to_openxr_pose(camera):
    c2w_render = np.eye(4, dtype=np.float32)
    c2w_render[:3, :3] = np.asarray(camera["rotation"], dtype=np.float32)
    c2w_render[:3, 3] = np.asarray(camera["position"], dtype=np.float32)

    c2w_openxr = c2w_render @ OPENXR_CAMERA_FROM_RENDER_CAMERA
    quat = _rotation_matrix_to_quaternion_xyzw(c2w_openxr[:3, :3])
    return {
        "position": c2w_openxr[:3, 3].tolist(),
        "orientation_xyzw": quat.tolist(),
    }


def _make_symmetric_fov(camera, cx, cy):
    width = float(camera["width"])
    height = float(camera["height"])
    fx = float(camera["fx"])
    fy = float(camera["fy"])

    return {
        "angle_left": float(-math.atan(cx / fx)),
        "angle_right": float(math.atan((width - cx) / fx)),
        "angle_up": float(math.atan(cy / fy)),
        "angle_down": float(-math.atan((height - cy) / fy)),
    }


def main():
    parser = argparse.ArgumentParser(description="Create an XR replay frame from an existing training/test camera.")
    parser.add_argument("--cameras_json", required=True, type=str)
    parser.add_argument("--camera_name", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--ipd", default=0.0, type=float, help="Eye separation in scene units. Use 0.0 to validate the bridge first.")
    parser.add_argument("--timestamp_ns", default=0, type=int)
    args = parser.parse_args()

    cameras = _load_cameras(args.cameras_json)
    camera = _find_camera(cameras, args.camera_name)

    width = int(camera["width"])
    height = int(camera["height"])
    cx = width * 0.5
    cy = height * 0.5
    pose = _camera_to_openxr_pose(camera)
    fov = _make_symmetric_fov(camera, cx=cx, cy=cy)

    rotation = np.asarray(camera["rotation"], dtype=np.float32)
    right_dir = rotation[:, 0]
    half_ipd = float(args.ipd) * 0.5
    base_position = np.asarray(pose["position"], dtype=np.float32)

    left_pose = {
        "position": (base_position - half_ipd * right_dir).tolist(),
        "orientation_xyzw": pose["orientation_xyzw"],
    }
    right_pose = {
        "position": (base_position + half_ipd * right_dir).tolist(),
        "orientation_xyzw": pose["orientation_xyzw"],
    }

    payload = {
        "frames": [
            {
                "frame_id": 0,
                "timestamp_ns": int(args.timestamp_ns),
                "views": {
                    "left": {
                        "pose": left_pose,
                        "fov": fov,
                        "image_rect": {"width": width, "height": height},
                    },
                    "right": {
                        "pose": right_pose,
                        "fov": fov,
                        "image_rect": {"width": width, "height": height},
                    },
                },
            }
        ]
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote XR replay frame for camera '{camera['img_name']}' to {args.output}")
    print(f"Camera position: {camera['position']}")
    print(f"fx/fy: {camera['fx']}, {camera['fy']}")
    print(f"resolution: {width}x{height}")


if __name__ == "__main__":
    main()
