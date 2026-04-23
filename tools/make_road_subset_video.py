import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class ClipSpec:
    name: str
    input_dir: Path
    start: int
    end: int


DEFAULT_CLIPS = [
    ClipSpec(
        name="street",
        input_dir=Path(
            "outputs/horizongs/real/road_subset/fine/road_subset_smooth_00056_00110/ours_40000/street/renders"
        ),
        start=0,
        end=216,
    ),
    ClipSpec(
        name="aerial",
        input_dir=Path(
            "outputs/horizongs/real/road_subset/fine/demo_orbit/ours_40000/aerial/renders"
        ),
        start=0,
        end=120,
    ),
]


def parse_size(size_text: str) -> tuple[int, int]:
    try:
        width_text, height_text = size_text.lower().split("x", 1)
        width = int(width_text)
        height = int(height_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid size '{size_text}'. Expected WIDTHxHEIGHT, e.g. 2560x1440."
        ) from exc

    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Video size must be positive.")
    return width, height


def resolve_output_path(output: str | None) -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    if output:
        return Path(output).expanduser().resolve()
    return repo_root / "road_subset_preview.mp4"


def collect_clip_frames(spec: ClipSpec) -> list[Path]:
    if not spec.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {spec.input_dir}")

    frames: list[Path] = []
    missing_indices: list[int] = []
    for index in range(spec.start, spec.end + 1):
        frame_path = spec.input_dir / f"{index:05d}.png"
        if frame_path.is_file():
            frames.append(frame_path)
        else:
            missing_indices.append(index)

    if missing_indices:
        preview = ", ".join(str(index) for index in missing_indices[:10])
        if len(missing_indices) > 10:
            preview += ", ..."
        raise FileNotFoundError(
            f"Missing {len(missing_indices)} frames in {spec.input_dir}: {preview}"
        )

    return frames


def read_frame(frame_path: Path) -> np.ndarray:
    frame = cv2.imread(str(frame_path))
    if frame is None:
        raise RuntimeError(f"Failed to read frame: {frame_path}")
    return frame


def build_resampled_sequence(frames: list[Path], output_frame_count: int) -> list[Path]:
    if output_frame_count <= 0:
        raise ValueError("Each clip must contribute at least one frame.")
    if not frames:
        raise ValueError("Cannot resample an empty clip.")
    if len(frames) == 1:
        return [frames[0]] * output_frame_count

    sample_positions = np.linspace(0, len(frames) - 1, num=output_frame_count)
    sample_indices = np.rint(sample_positions).astype(int)
    return [frames[index] for index in sample_indices]


def fit_with_padding(frame: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    target_w, target_h = target_size
    src_h, src_w = frame.shape[:2]

    scale = min(target_w / src_w, target_h / src_h)
    resized_w = max(1, int(round(src_w * scale)))
    resized_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    offset_x = (target_w - resized_w) // 2
    offset_y = (target_h - resized_h) // 2
    canvas[offset_y : offset_y + resized_h, offset_x : offset_x + resized_w] = resized
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Make a single video from the road_subset street and aerial render frames."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output video path. Defaults to <repo_root>/road_subset_preview.mp4",
    )
    parser.add_argument(
        "--target-seconds",
        type=float,
        default=10.0,
        help="Approximate total video length when --street-seconds and --aerial-seconds are not set.",
    )
    parser.add_argument(
        "--street-seconds",
        type=float,
        default=None,
        help="Target duration for the street clip. Must be used together with --aerial-seconds.",
    )
    parser.add_argument(
        "--aerial-seconds",
        type=float,
        default=None,
        help="Target duration for the aerial clip. Must be used together with --street-seconds.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Use a fixed FPS instead of deriving it from the requested total duration.",
    )
    parser.add_argument(
        "--size",
        type=parse_size,
        default=None,
        help="Output size as WIDTHxHEIGHT. Defaults to the first selected frame size.",
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="mp4v",
        help="FourCC codec for cv2.VideoWriter.",
    )
    args = parser.parse_args()

    clip_frames = {spec.name: collect_clip_frames(spec) for spec in DEFAULT_CLIPS}
    original_total_frames = sum(len(frames) for frames in clip_frames.values())
    if original_total_frames == 0:
        raise RuntimeError("No frames were selected.")

    use_custom_clip_durations = (
        args.street_seconds is not None or args.aerial_seconds is not None
    )
    if use_custom_clip_durations and (
        args.street_seconds is None or args.aerial_seconds is None
    ):
        raise ValueError(
            "--street-seconds and --aerial-seconds must be provided together."
        )

    if use_custom_clip_durations:
        if args.street_seconds <= 0 or args.aerial_seconds <= 0:
            raise ValueError("Clip durations must be positive.")
        total_seconds = args.street_seconds + args.aerial_seconds
    else:
        if args.target_seconds <= 0:
            raise ValueError("--target-seconds must be positive.")
        total_seconds = args.target_seconds

    fps = args.fps if args.fps is not None else original_total_frames / total_seconds
    if fps <= 0:
        raise ValueError("FPS must be positive.")

    if use_custom_clip_durations:
        requested_seconds = {
            "street": args.street_seconds,
            "aerial": args.aerial_seconds,
        }
        all_frames: list[Path] = []
        for spec in DEFAULT_CLIPS:
            clip_output_frames = max(1, int(round(requested_seconds[spec.name] * fps)))
            all_frames.extend(
                build_resampled_sequence(clip_frames[spec.name], clip_output_frames)
            )
    else:
        all_frames = []
        for spec in DEFAULT_CLIPS:
            all_frames.extend(clip_frames[spec.name])

    first_frame = read_frame(all_frames[0])
    target_size = args.size or (first_frame.shape[1], first_frame.shape[0])

    output_path = resolve_output_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, target_size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer: {output_path}")

    written_frames = 0
    try:
        for frame_path in all_frames:
            frame = read_frame(frame_path)
            frame = fit_with_padding(frame, target_size)
            writer.write(frame)
            written_frames += 1
    finally:
        writer.release()

    if written_frames == 0:
        raise RuntimeError("No frames were written to the output video.")

    duration = written_frames / fps
    print(f"Saved video to: {output_path}")
    print(f"Frames: {written_frames}")
    print(f"FPS: {fps:.3f}")
    print(f"Duration: {duration:.3f} seconds")
    print(f"Resolution: {target_size[0]}x{target_size[1]}")
    if use_custom_clip_durations:
        print(f"Street clip target duration: {args.street_seconds:.3f} seconds")
        print(f"Aerial clip target duration: {args.aerial_seconds:.3f} seconds")


if __name__ == "__main__":
    main()
