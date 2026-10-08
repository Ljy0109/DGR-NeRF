import argparse
import os
import sys
from pathlib import Path

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="从 txt 文件中读取 .tif 路径，批量转换为 .png"
    )
    parser.add_argument("txt_path", type=str, help="包含 .tif 路径的 txt 文件")
    parser.add_argument("output_dir", type=str, help="输出 .png 的目录")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="若输出 png 已存在则覆盖（默认跳过已存在的文件）"
    )
    parser.add_argument(
        "--keep-structure", action="store_true",
        help="按原 tif 文件的父目录名在输出目录下创建子目录（防止同名冲突）"
    )
    parser.add_argument(
        "--mode", type=str, default="RGB",
        help="保存时的颜色模式，默认 RGB（可选 RGBA、L 等）"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="只打印最终统计，不打印每个文件的转换日志"
    )
    return parser.parse_args()


def convert_tif_to_png_from_txt(
    txt_path: Path,
    output_dir: Path,
    overwrite: bool = False,
    keep_structure: bool = False,
    mode: str = "RGB",
    quiet: bool = False,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # 读取路径列表并过滤空行
    with txt_path.open("r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    n_total = len(lines)
    n_ok = n_skip_missing = n_skip_exists = n_fail = 0

    for tif_path_str in lines:
        tif_path = Path(tif_path_str)

        if not tif_path.exists():
            print(f"[跳过] 文件不存在: {tif_path}")
            n_skip_missing += 1
            continue

        # 决定输出子目录
        if keep_structure:
            sub_dir = output_dir / tif_path.parent.name
        else:
            sub_dir = output_dir
        sub_dir.mkdir(parents=True, exist_ok=True)

        png_path = sub_dir / (tif_path.stem + ".png")

        if png_path.exists() and not overwrite:
            if not quiet:
                print(f"[跳过] 已存在: {png_path}")
            n_skip_exists += 1
            continue

        try:
            with Image.open(tif_path) as img:
                img.convert(mode).save(png_path)
            if not quiet:
                print(f"[成功] {tif_path} → {png_path}")
            n_ok += 1
        except Exception as e:
            print(f"[失败] 转换 {tif_path} 时出错: {e}")
            n_fail += 1

    # 统计
    print("=" * 50)
    print(f"总计 : {n_total}")
    print(f"成功 : {n_ok}")
    print(f"跳过 (文件不存在): {n_skip_missing}")
    print(f"跳过 (已存在)    : {n_skip_exists}")
    print(f"失败 : {n_fail}")

    if n_fail > 0:
        sys.exit(1)


def main():
    args = parse_args()

    txt_path = Path(args.txt_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not txt_path.is_file():
        print(f"[错误] txt 文件不存在: {txt_path}")
        sys.exit(1)

    convert_tif_to_png_from_txt(
        txt_path=txt_path,
        output_dir=output_dir,
        overwrite=args.overwrite,
        keep_structure=args.keep_structure,
        mode=args.mode,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()