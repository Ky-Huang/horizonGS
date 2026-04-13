import argparse
from pathlib import Path
from PIL import Image

# 支持的图片后缀
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def resize_images_bilinear_2x_keep_structure(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 递归查找所有图片
    image_files = [
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

    if not image_files:
        print(f"在目录中没有找到支持的图片文件: {input_dir}")
        return

    print(f"找到 {len(image_files)} 张图片，开始进行 2 倍双线性上采样...")

    for img_path in image_files:
        try:
            with Image.open(img_path) as img:
                if img.mode not in ("RGB", "RGBA", "L"):
                    img = img.convert("RGB")

                w, h = img.size
                new_size = (w * 2, h * 2)
                resized_img = img.resize(new_size, Image.BILINEAR)

                # 计算相对路径，并拼接到输出目录
                relative_path = img_path.relative_to(input_dir)
                save_path = output_dir / relative_path

                # 创建对应子目录
                save_path.parent.mkdir(parents=True, exist_ok=True)

                # JPEG 不支持透明通道
                if save_path.suffix.lower() in {".jpg", ".jpeg"} and resized_img.mode == "RGBA":
                    resized_img = resized_img.convert("RGB")

                resized_img.save(save_path)
                print(f"已保存: {save_path} | {w}x{h} -> {new_size[0]}x{new_size[1]}")

        except Exception as e:
            print(f"处理失败: {img_path} | 错误: {e}")

    print("全部处理完成。")


def main():
    parser = argparse.ArgumentParser(description="递归处理目录中的所有图片，保持目录结构并做 2 倍双线性上采样")
    parser.add_argument("--input_dir", type=str, required=True, help="输入图片目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出图片目录")
    args = parser.parse_args()

    resize_images_bilinear_2x_keep_structure(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()