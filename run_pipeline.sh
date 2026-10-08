set -euo pipefail 

DFC_IMG_DIR="/root/mycode/DGR/DFC2019"
OUT_DIR="$DFC_IMG_DIR/outputs"
SENCE_id="JAX_068"

python DFC2019_Preprocess_single.py "$DFC_IMG_DIR" "$OUT_DIR" "$SENCE_id"

python create_txt.py "$OUT_DIR/$SENCE_id" "$OUT_DIR/$SENCE_id"

python tif2png.py "$OUT_DIR/$SENCE_id/tif2png.txt" "$OUT_DIR/$SENCE_id/tif2png"

python run_depth_anything.py --img-path "$OUT_DIR/$SENCE_id/rgb2depth.txt" --outdir "$OUT_DIR/$SENCE_id/vis_depth"

python ecef2utm.py "$OUT_DIR/$SENCE_id/ba_files/ba_params/pts3d.npy"

python main.py --case "$SENCE_id"