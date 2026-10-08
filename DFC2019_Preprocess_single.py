import os
import sys
import cv2
import json
import glob
import rpcm
import srtm4
import shutil
import rasterio
import argparse
import numpy as np
from skimage import io
from bundle_adjust import loader
from bundle_adjust import geo_utils
from bundle_adjust.cam_utils import SatelliteImage
from bundle_adjust.ba_pipeline import BundleAdjustmentPipeline


def read_lonlat_roi(dsm_path):
    # 读取给定 dsm_path（数字表面模型，DSM）对应的地理范围，并转换成经纬度格式。
    if os.path.basename(dsm_path).startswith('OMA'):
        zonestring = "15"  
    elif os.path.basename(dsm_path).startswith('JAX'):
        zonestring = "17"  
    else:
        raise ValueError('Unknown sence id: {}'.format(os.path.basename(dsm_path)))
    roi = np.loadtxt(dsm_path.replace('tif', 'txt'))
    xoff, yoff, xsize, ysize, resolution = roi[0], roi[1], int(roi[2]), int(roi[2]), roi[3]
    xmin, ymin, xmax, ymax = xoff, yoff + ysize * resolution, xoff + xsize * resolution, yoff  
    easts = [xmin, xmin, xmax, xmax, xmin] 
    norths = [ymin, ymax, ymax, ymin, ymin] 
    lons, lats = geo_utils.lonlat_from_utm(easts, norths, zonestring)
    lonlat_bbx = geo_utils.geojson_polygon(np.vstack((lons, lats)).T)  
    return lonlat_bbx

def crop_tiff_lonlat_roi(geotiff_path, output_path, lonlat_aoi, minheight):
    with rasterio.open(geotiff_path, 'r') as src:
        profile = src.profile   
        tags = src.tags()   
    crop, x, y = rpcm.utils.crop_aoi(geotiff_path, lonlat_aoi, z=minheight)
    rpc = rpcm.rpc_from_geotiff(geotiff_path)
    rpc.row_offset -= y
    rpc.col_offset -= x
    profile["height"] = crop.shape[1]
    profile["width"] = crop.shape[2]

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(crop) 
        dst.update_tags(**tags) 
        dst.update_tags(ns='RPC', **rpc.to_geotiff_dict())  

def crop_roi_tif(input_paths, out_dir, dsm_info_path):
    roi_lonlat = read_lonlat_roi(dsm_info_path)
    with rasterio.open(dsm_info_path.replace('txt', 'tif'), 'r') as src:
        data = src.read() 
    minheight = int(np.round(data.min() - 1))

    for img_path in input_paths:
        out_path = os.path.join(out_dir, os.path.basename(img_path))
        crop_tiff_lonlat_roi(img_path, out_path, roi_lonlat, minheight)

def get_image_lonlat_aoi(rpc, h, w):
    z = srtm4.srtm4(rpc.lon_offset, rpc.lat_offset)
    cols, rows, alts = [0,w,w,0], [0,0,h,h], [z]*4
    lons, lats = rpc.localization(cols, rows, alts)
    lonlat_coords = np.vstack((lons, lats)).T
    geojson_polygon = {"coordinates": [lonlat_coords.tolist()], "type": "Polygon"}
    x_c = lons.min() + (lons.max() - lons.min())/2
    y_c = lats.min() + (lats.max() - lats.min())/2
    geojson_polygon["center"] = [x_c, y_c]
    return geojson_polygon

def bundle_adjustment(img_dir, output_dir):
    images = sorted(glob.glob(img_dir + "/*_RGB.tif"))
    rpcs = [rpcm.rpc_from_geotiff(p) for p in images]
    input_images = [SatelliteImage(fn, rpc) for fn, rpc in zip(images, rpcs)]

    ba_input_data = {}
    ba_input_data['in_dir'] = img_dir
    ba_input_data['out_dir'] = os.path.join(output_dir, "ba_files")
    ba_input_data['images'] = input_images
    print('Input data set!\n')

    os.makedirs(ba_input_data['out_dir'], exist_ok=True)
    path_to_log_file = "{}/bundle_adjust.log".format(ba_input_data['out_dir'])
    print("Running bundle adjustment for RPC model refinement ...")
    print("Path to log file: {}".format(path_to_log_file))
    log_file = open(path_to_log_file, "w+")
    sys.stdout = log_file
    sys.stderr = log_file
    tracks_config = {'FT_reset': False, 'FT_save': True, 'FT_sift_detection': 's2p', 'FT_sift_matching': 'epipolar_based'}
    ba_extra = {"cam_model": "rpc"}
    ba_pipeline = BundleAdjustmentPipeline(ba_input_data, tracks_config=tracks_config, extra_ba_config=ba_extra)
    ba_pipeline.run()
    sys.stderr = sys.__stderr__
    sys.stdout = sys.__stdout__
    log_file.close()
    print("... done !")
    print("Path to output files: {}".format(ba_input_data['out_dir']))

    ba_params_dir = os.path.join(ba_pipeline.out_dir, "ba_params")
    os.makedirs(ba_params_dir, exist_ok=True)
    np.save(os.path.join(ba_params_dir, "pts_ind.npy"), ba_pipeline.ba_params.pts_ind)
    np.save(os.path.join(ba_params_dir, "cam_ind.npy"), ba_pipeline.ba_params.cam_ind)
    np.save(os.path.join(ba_params_dir, "pts3d.npy"), ba_pipeline.ba_params.pts3d_ba - ba_pipeline.global_transform)
    np.save(os.path.join(ba_params_dir, "pts2d.npy"), ba_pipeline.ba_params.pts2d)
    fnames_in_use = [ba_pipeline.images[idx].geotiff_path for idx in ba_pipeline.ba_params.cam_prev_indices]
    loader.save_list_of_paths(os.path.join(ba_params_dir, "geotiff_paths.txt"), fnames_in_use)

def create_dataset(sence_dir, sence_id, DFC_img_dir, use_ba=True):
    path_to_dsm = os.path.join(sence_dir, "{}_DSM.tif".format(sence_id))
    if use_ba:
        geotiff_paths = sorted(glob.glob(os.path.join(sence_dir, sence_id + '_*_RGB.tif')))
        ba_geotiff_basenames = [os.path.basename(x) for x in geotiff_paths]
        ba_kps_pts3d_ind = np.load(os.path.join(sence_dir, "ba_files/ba_params/pts_ind.npy"))
        ba_kps_cam_ind = np.load(os.path.join(sence_dir, "ba_files/ba_params/cam_ind.npy"))
        ba_kps_pts2d = np.load(os.path.join(sence_dir, "ba_files/ba_params/pts2d.npy"))
    else:
        geotiff_paths = sorted(glob.glob(os.path.join(sence_dir, sence_id + '_*_RGB.tif')))

    for rgb_p in geotiff_paths: 
        img_id = os.path.basename(rgb_p)[:-4]
        d = {}
        d["img"] = os.path.basename(rgb_p) 

        src = rasterio.open(rgb_p)
        d["height"] = int(src.meta["height"])
        d["width"] = int(src.meta["width"])
        original_rpc = rpcm.RPCModel(src.tags(ns='RPC'), dict_format="geotiff")

        with open(os.path.join(DFC_img_dir, "Track3-Metadata", img_id[:3], "{}.IMD".format(img_id[9:11])), 'r') as f:
            for line in f:
                if "meanSunAz" in line:
                    d["sun_azimuth"] = float(line.split("=")[1].split(";")[0].strip())
                if "meanSunEl" in line:
                    d["sun_elevation"] = float(line.split("=")[1].split(";")[0].strip())

        d["geojson"] = get_image_lonlat_aoi(original_rpc, d["height"], d["width"])

        src = rasterio.open(path_to_dsm)
        dsm = src.read()[0, :, :]
        d["min_alt"] = int(np.round(dsm.min() - 1))
        d["max_alt"] = int(np.round(dsm.max() + 1))

        if use_ba:
            rpc_path = os.path.join(sence_dir, "ba_files/rpcs_adj/{}.rpc_adj".format(img_id))
            d["rpc"] = rpcm.rpc_from_rpc_file(rpc_path).__dict__
            d["rpc_ori"] = original_rpc.__dict__
        else:
            d["rpc"] = original_rpc.__dict__

        with open(os.path.join(sence_dir, "{}.json".format(img_id)), "w") as f:
            json.dump(d, f, indent=2)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dfc_img_dir", type=str
    )
    parser.add_argument(
        "out_dir", type=str
    )
    parser.add_argument(
        "sence_id", type=str
    )
    return parser.parse_args()

if __name__ == '__main__':
    arg = parse_args()
    DFC_img_dir = arg.dfc_img_dir
    out_dir = arg.out_dir
    
    sence_id = arg.sence_id
    sence_dict = {}
    sence_dict['dsm'] = os.path.join(DFC_img_dir, 'Track3-Truth', sence_id + '_DSM.tif')
    sence_dict['dsm_info'] = os.path.join(DFC_img_dir, 'Track3-Truth', sence_id + '_DSM.txt')
    sence_dict['img'] = sorted(glob.glob(os.path.join(DFC_img_dir, 'Track3-RGB-*', sence_id + '_*_RGB.tif')))
    
    sence_dir = os.path.join(out_dir, sence_id)
    os.makedirs(sence_dir, exist_ok=True)

    crop_roi_tif(sence_dict['img'], sence_dir, sence_dict['dsm_info'])
    shutil.copy(sence_dict['dsm'], sence_dir)
    # gt = io.imread(sence_dict['dsm'])
    # cv2.imwrite(os.path.join(sence_dir, sence_id + '_DSM.tif'), gt)
    shutil.copy(sence_dict['dsm_info'], sence_dir)
    bundle_adjustment(sence_dir, sence_dir)
    create_dataset(sence_dir, sence_id, DFC_img_dir)
