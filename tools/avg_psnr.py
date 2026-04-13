import os
import math
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS


def load_image_as_rgb_array(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.float64)


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    if img1.shape != img2.shape:
        raise ValueError(f"图片尺寸不一致: {img1.shape} vs {img2.shape}")

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")

    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr


def collect_image_files(root: Path):
    return sorted(
        [p for p in root.rglob("*") if p.is_file() and is_image_file(p)]
    )


def main():
    parser = argparse.ArgumentParser(description="计算两个目录对应图片的平均 PSNR")
    parser.add_argument("dir1", type=str, help="目录1")
    parser.add_argument("dir2", type=str, help="目录2")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="严格模式：如果有任意图片缺失或尺寸不一致，直接报错退出",
    )
    args = parser.parse_args()

    dir1 = Path(args.dir1).resolve()
    dir2 = Path(args.dir2).resolve()

    if not dir1.is_dir():
        raise FileNotFoundError(f"目录不存在: {dir1}")
    if not dir2.is_dir():
        raise FileNotFoundError(f"目录不存在: {dir2}")

    files1 = collect_image_files(dir1)
    if not files1:
        raise RuntimeError(f"目录中没有找到图片: {dir1}")

    psnr_list = []
    skipped = []

    for img1_path in files1:
        rel_path = img1_path.relative_to(dir1)
        img2_path = dir2 / rel_path

        if not img2_path.exists():
            msg = f"对应图片不存在: {img2_path}"
            if args.strict:
                raise FileNotFoundError(msg)
            skipped.append(str(rel_path))
            print(f"[跳过] {msg}")
            continue

        try:
            img1 = load_image_as_rgb_array(img1_path)
            img2 = load_image_as_rgb_array(img2_path)
            psnr = compute_psnr(img1, img2)
            psnr_list.append(psnr)
            print(f"{rel_path}: PSNR = {psnr:.4f} dB")
        except Exception as e:
            msg = f"{rel_path}: {e}"
            if args.strict:
                raise RuntimeError(msg)
            skipped.append(str(rel_path))
            print(f"[跳过] {msg}")

    if not psnr_list:
        raise RuntimeError("没有成功计算任何图片对的 PSNR")

    finite_psnr = [x for x in psnr_list if math.isfinite(x)]
    inf_count = len(psnr_list) - len(finite_psnr)

    if finite_psnr:
        avg_psnr = sum(finite_psnr) / len(finite_psnr)
        print("\n========== 结果 ==========")
        print(f"成功匹配图片对数: {len(psnr_list)}")
        print(f"其中完全相同（PSNR=inf）数量: {inf_count}")
        print(f"有限 PSNR 的平均值: {avg_psnr:.4f} dB")
    else:
        print("\n========== 结果 ==========")
        print(f"成功匹配图片对数: {len(psnr_list)}")
        print(f"所有图片都完全相同，平均 PSNR = inf")

    if skipped:
        print(f"跳过数量: {len(skipped)}")


if __name__ == "__main__":
    main()