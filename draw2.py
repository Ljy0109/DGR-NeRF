#!/usr/bin/env python3
import os
import argparse
from glob import glob
import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def read_first_band(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
    return arr

def compute_global_vmin_vmax(paths, method="minmax", lower_pct=2.0, upper_pct=98.0, nodata=None):
    vals = []
    for p in paths:
        arr = read_first_band(p)
        if nodata is None:
            mask = ~np.isnan(arr)
        else:
            mask = arr != nodata
        if np.any(mask):
            vals.append(arr[mask].ravel())
    if len(vals) == 0:
        raise ValueError("No valid pixels found in inputs.")
    allv = np.concatenate(vals)
    if method == "minmax":
        return float(np.nanmin(allv)), float(np.nanmax(allv))
    elif method == "percentile":
        lo = float(np.nanpercentile(allv, lower_pct))
        hi = float(np.nanpercentile(allv, upper_pct))
        return lo, hi
    else:
        raise ValueError("Unknown method")

def save_colored_png_with_colorbar_minmax(array, out_png, cmap_name, vmin, vmax, colorbar_label=None, dpi=200):
    """
    array: 2D numpy (float), NaN should already be filled by caller if desired.
    out_png: output path (png)
    cmap_name: matplotlib colormap name
    vmin/vmax: normalization bounds
    Saves PNG with vertical colorbar on the right that only shows the min and max numeric labels.
    """
    # array assumed to have no NaN (or caller can pass with NaN, but we'll handle)
    arr = array.copy()
    # If any NaNs remain, mask them for plotting (but generally we replaced them before calling)
    masked = np.ma.masked_invalid(arr)

    cmap = plt.get_cmap(cmap_name).copy()
    # If masked exists, set bad -> transparent; otherwise no effect
    try:
        cmap.set_bad(color=(0,0,0,0))
    except Exception:
        pass

    h, w = arr.shape
    width_inch = max(1.0, w / float(dpi))
    height_inch = max(1.0, h / float(dpi))
    fig = plt.figure(figsize=(width_inch * 1.15, height_inch), dpi=dpi)
    ax_img = fig.add_axes([0.00, 0.05, 0.85, 0.90])
    ax_cb = fig.add_axes([0.88, 0.10, 0.03, 0.8])

    im = ax_img.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax)
    ax_img.axis('off')

    # colorbar，只显示最小/最大两个刻度
    cbar = plt.colorbar(im, cax=ax_cb, orientation='vertical')
    try:
        cbar.set_ticks([vmin, vmax])
        # 自动决定格式
        if float(vmin).is_integer() and float(vmax).is_integer():
            fmt = "{:.0f}"
        else:
            fmt = "{:.2f}"
        cbar.ax.set_yticklabels([fmt.format(vmin), fmt.format(vmax)])
    except Exception:
        pass

    if colorbar_label:
        cbar.set_label(colorbar_label, rotation=90, va='bottom')

    # 透明背景保存（注意：若所有 NaN 已被替换，PNG 背景不会是透明的图像区域）
    fig.patch.set_alpha(0.0)
    ax_img.patch.set_alpha(0.0)
    plt.savefig(out_png, transparent=True, bbox_inches='tight', pad_inches=0.01)
    plt.close(fig)

def collect_input_files(input_dir=None, input_pattern=None, files_list=None):
    files = []
    if files_list:
        for l in files_list:
            if os.path.exists(l):
                files.append(l)
            else:
                files.extend(sorted(glob(l)))
    if input_dir:
        # 支持常见 tif 扩展名
        files.extend(sorted(glob(os.path.join(input_dir, "*.[tT][iI][fF]"))))
        files.extend(sorted(glob(os.path.join(input_dir, "*.tif"))))
        files.extend(sorted(glob(os.path.join(input_dir, "*.png"))))
        files.extend(sorted(glob(os.path.join(input_dir, "*.jpg"))))
        files.extend(sorted(glob(os.path.join(input_dir, "*.jpeg"))))
    if input_pattern:
        files.extend(sorted(glob(input_pattern)))
    seen = set(); out = []
    for f in files:
        if f not in seen:
            seen.add(f); out.append(f)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", help="目录，包含多个 DSM GeoTIFF (单波段)")
    p.add_argument("--input_pattern", help="glob 模式，例如 'dataset/*/*_dsm.tif'")
    p.add_argument("--files", nargs="*", help="可选：直接列出文件路径或 glob 模式")
    p.add_argument("--output_dir", help="输出目录")
    p.add_argument("--cmap", default="jet",
                   help="matplotlib colormap 名称（例如 viridis, jet, plasma, turbo, magma, inferno, seismic）")
    p.add_argument("--vmin", type=float, default=None, help="显式的最小值（若不指定则自动计算）")
    p.add_argument("--vmax", type=float, default=None, help="显式的最大值（若不指定则自动计算）")
    p.add_argument("--percentile", nargs=2, type=float, metavar=('LOW', 'HIGH'), default=None,
                   help="若指定，用全体像素的百分位来决定 vmin/vmax（例如 --percentile 2 98）")
    p.add_argument("--colorbar_label", default="elevation (m)", help="色柱标签")
    p.add_argument("--dpi", type=int, default=200, help="输出图像 DPI（影响保存尺寸与清晰度）")
    p.add_argument("--force", action="store_true", help="覆盖已存在的输出")
    args = p.parse_args()

    args.input_dir = "draw/view"
    args.output_dir = "draw/output/view"
    args.force = True
    args.percentile = (0, 100)

    os.makedirs(args.output_dir, exist_ok=True)

    files = collect_input_files(args.input_dir, args.input_pattern, args.files)
    if len(files) == 0:
        raise SystemExit("未找到任何输入文件，请检查 input_dir / input_pattern / files 参数。")

    print(f"找到 {len(files)} 文件，示例：{os.path.basename(files[0])}")

    # 先查找是否有名为 gt 的文件（文件名包含 "gt"，不区分大小写）
    gt_file = None
    for f in files:
        bn = os.path.basename(f).lower()
        if "gt" in bn:
            gt_file = f
            break

    # 如果找到了 gt 文件，直接用 gt 的有效像素范围作为 vmin/vmax（覆盖其它设置）
    if gt_file is not None:
        print("找到 gt 文件:", os.path.basename(gt_file))
        gt_arr = read_first_band(gt_file)
        valid_mask = ~np.isnan(gt_arr)
        if not np.any(valid_mask):
            raise SystemExit(f"gt 文件 {gt_file} 没有有效像素。")
        gt_vmin = float(np.nanmin(gt_arr[valid_mask]))
        gt_vmax = float(np.nanmax(gt_arr[valid_mask]))
        vmin, vmax = gt_vmin, gt_vmax
        print(f"使用 gt 的范围做色柱: vmin={vmin}, vmax={vmax}")
    else:
        # 否则使用命令行或全局计算
        if args.vmin is not None and args.vmax is not None:
            vmin, vmax = args.vmin, args.vmax
            print(f"使用显式传入的 vmin/vmax -> {vmin}, {vmax}")
        else:
            if args.percentile:
                lo, hi = float(args.percentile[0]), float(args.percentile[1])
                vmin, vmax = compute_global_vmin_vmax(files, method="percentile", lower_pct=lo, upper_pct=hi)
                print(f"使用 percentiles {lo}/{hi} 做 vmin/vmax -> {vmin:.4f}, {vmax:.4f}")
            else:
                vmin, vmax = compute_global_vmin_vmax(files, method="minmax")
                print(f"使用全局 min/max 做 vmin/vmax -> {vmin:.4f}, {vmax:.4f}")

    # 逐个处理并保存：先把 NaN 替换为该图像的最小值，再根据是否存在 gt 对值做裁剪
    for fin in files:
        try:
            arr = read_first_band(fin)
        except Exception as e:
            print(f"读取失败：{fin} -> {e}")
            continue
        base = os.path.splitext(os.path.basename(fin))[0]
        out_png = os.path.join(args.output_dir, f"{base}.png")
        if os.path.exists(out_png) and not args.force:
            print(f"已存在，跳过: {out_png}")
            continue

        print(f"处理 {os.path.basename(fin)} -> {os.path.basename(out_png)}")

        # ------------------ 把 NaN 替换为该图像的最小值 ------------------
        nan_mask = np.isnan(arr)
        if np.all(nan_mask):
            print(f"警告：文件 {fin} 全部为 NaN，跳过。")
            continue
        # 只计算有效像素的最小值
        valid_min = float(np.nanmin(arr[~nan_mask]))
        # 将 NaN 替换为该图像的最小值（原地或拷贝都可以）
        arr_filled = arr.copy()
        arr_filled[nan_mask] = valid_min
        # -------------------------------------------------------------------------

        # 如果存在 gt，则把当前图像裁剪到 gt 的范围（超出部分变为边界值）
        if gt_file is not None and fin != gt_file:
            arr_to_plot = np.clip(arr_filled, vmin, vmax)
        else:
            # gt 自己或没有 gt 时，使用全局 vmin/vmax（已经计算好）
            arr_to_plot = arr_filled.copy()
            # 如果没有 gt，但用户传入了 vmin/vmax，仍然按照这些值在显示中限制色条（但不需要 pre-clamp）
            if gt_file is None:
                arr_to_plot = np.clip(arr_to_plot, vmin, vmax)

        # 保存（使用 vmin/vmax）
        save_colored_png_with_colorbar_minmax(arr_to_plot, out_png,
                                              cmap_name=args.cmap,
                                              vmin=vmin, vmax=vmax,
                                              colorbar_label=args.colorbar_label,
                                              dpi=args.dpi)
    print("全部完成。")
