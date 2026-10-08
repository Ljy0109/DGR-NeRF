import argparse
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="扫描文件夹，生成 tif2png / rgb2depth / train_depth / train 四个清单文件"
    )
    parser.add_argument(
        "folder", type=str,
        help="要扫描的文件夹（绝对路径或相对路径）"
    )
    parser.add_argument(
        "output_dir", type=str, nargs="?", default=None,
        help="输出目录，默认写到当前工作目录"
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="递归扫描子目录（默认只扫描当前层）"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="不打印统计信息"
    )
    return parser.parse_args()


def collect_files(folder: Path, recursive: bool = False):
    pattern = "**/" if recursive else ""
    tif_files = sorted(folder.glob(f"{pattern}*_RGB.tif"))
    json_files = sorted(folder.glob(f"{pattern}*_RGB.json"))
    return tif_files, json_files


def write_lines(filepath: Path, lines):
    if not lines:
        filepath.write_text("", encoding="utf-8")
        return
    filepath.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        print(f"[错误] 文件夹不存在或不是目录: {folder}")
        sys.exit(1)

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir else Path.cwd()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    tif_files, json_files = collect_files(folder, recursive=args.recursive)

    if not tif_files:
        print(f"[警告] 在 {folder} 中未找到 *_RGB.tif 文件")
    if not json_files:
        print(f"[警告] 在 {folder} 中未找到 *_RGB.json 文件")

    tif_paths = [str(p) for p in tif_files]
    write_lines(output_dir / "tif2png.txt", tif_paths)

    rgb2depth_paths = [
        str(p.parent / "tif2png" / (p.stem + ".png"))
        for p in tif_files
    ]
    write_lines(output_dir / "rgb2depth.txt", rgb2depth_paths)

    train_depth_paths = [
        "/vis_depth/" + p.stem + "_depth.png"
        for p in tif_files
    ]
    write_lines(output_dir / "train_depth.txt", train_depth_paths)

    json_paths = [str(p) for p in json_files]
    write_lines(output_dir / "train.txt", json_paths)

    if not args.quiet:
        print(f"[完成] 目录: {folder}")
        print(f"[完成] 输出: {output_dir}")
        print(f"  tif2png.txt     : {len(tif_paths)} 行")
        print(f"  rgb2depth.txt   : {len(rgb2depth_paths)} 行")
        print(f"  train_depth.txt : {len(train_depth_paths)} 行")
        print(f"  train.txt       : {len(json_paths)} 行")


if __name__ == "__main__":
    main()