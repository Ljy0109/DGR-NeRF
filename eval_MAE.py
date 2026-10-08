import rasterio
import numpy as np

def read_dsm_data(path):
    with rasterio.open(path) as src:
        dsm = src.read(1).astype(np.float32)
    return dsm

def compute_mae(dsm1, dsm2):
    # 掩码：只计算两个 DSM 都是有效数据的地方（排除 NaN 或 nodata）
    valid_mask = np.isfinite(dsm1) & np.isfinite(dsm2)
    if np.count_nonzero(valid_mask) == 0:
        raise ValueError("两个 DSM 没有重叠的有效区域！")
    diff = np.abs(dsm1[valid_mask] - dsm2[valid_mask]) 
    mae = np.mean(diff)  
    return mae

target_dsm_path = "exp/JAX_260/validations_fine/00100000_dsm.tif"
gt_dsm_path = "Track3/Track3-Truth/JAX_260_DSM.tif"

dsm = read_dsm_data(target_dsm_path)
gt_dsm = read_dsm_data(gt_dsm_path)

# 计算 MAE
mae = compute_mae(dsm, gt_dsm)
print(f"DSM 与 GT_DSM 的 MAE 为: {mae:.4f}")
