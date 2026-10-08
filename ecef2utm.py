import argparse
from pathlib import Path

import numpy as np
import utm
from pyproj import Transformer


def parse_args():
    parser = argparse.ArgumentParser(
        description="将 ECEF (EPSG:4978) 坐标 .npy 转换为 UTM 坐标 .npy"
    )
    parser.add_argument(
        "input", type=str,
        help="输入 .npy 文件路径（形状 (N, 3)，ECEF 坐标）"
    )
    parser.add_argument(
        "output", type=str, nargs="?", default=None,
        help="输出 .npy 文件路径（默认：与输入同目录，名为 utm_pts3d.npy）"
    )
    parser.add_argument(
        "--epsg-in", type=int, default=4978,
        help="输入坐标系 EPSG，默认 4978 (ECEF)"
    )
    parser.add_argument(
        "--epsg-geodetic", type=int, default=4326,
        help="中间地理坐标系 EPSG，默认 4326 (WGS84)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="不打印统计信息"
    )
    return parser.parse_args()


def ecef_to_utm(ecef_pts: np.ndarray, epsg_in: int, epsg_geodetic: int):
    transformer = Transformer.from_crs(
        f"epsg:{epsg_in}", f"epsg:{epsg_geodetic}", always_xy=True
    )
    lon, lat, alt = transformer.transform(
        ecef_pts[:, 0], ecef_pts[:, 1], ecef_pts[:, 2]
    )
    lon = np.asarray(lon)
    lat = np.asarray(lat)
    alt = np.asarray(alt)

    utm_results = [utm.from_latlon(lat[i], lon[i]) for i in range(len(lat))]
    easting = np.array([res[0] for res in utm_results])
    northing = np.array([res[1] for res in utm_results])
    zone_numbers = np.array([res[2] for res in utm_results])
    zone_letters = np.array([res[3] for res in utm_results])

    utm_pts = np.stack([easting, northing, alt], axis=1)
    return utm_pts, zone_numbers, zone_letters


def main():
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    if args.output is None:
        output_path = input_path.parent / "utm_pts3d.npy"
    else:
        output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ecef_pts = np.load(str(input_path))
    if ecef_pts.ndim != 2 or ecef_pts.shape[1] != 3:
        raise ValueError(
            f"期望输入形状为 (N, 3)，实际为 {ecef_pts.shape}"
        )

    utm_pts, zone_numbers, zone_letters = ecef_to_utm(
        ecef_pts, args.epsg_in, args.epsg_geodetic
    )

    np.save(str(output_path), utm_pts)

    if not args.quiet:
        print(f"输入: {input_path}")
        print(f"输出: {output_path}")
        print(f"转换完成，保存 {utm_pts.shape[0]} 个点到 {output_path.name}")
        if len(zone_numbers) > 0:
            n_show = min(5, len(zone_numbers))
            print(f"包含 UTM 区域号 (前{n_show}个):",
                  zone_numbers[:n_show], zone_letters[:n_show])
            unique_zones = sorted(set(zone_numbers.tolist()))
            print(f"涉及 UTM 区域号: {unique_zones}")


if __name__ == "__main__":
    main()