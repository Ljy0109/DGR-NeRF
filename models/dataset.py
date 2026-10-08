import os
import utm
import json
import rpcm
import torch
import numpy as np
import models.sat_utils as sat_utils
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Dataset:
    def __init__(self, conf):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf
        self.train = True
        self.img_downscale = conf.get_float('img_downscale')
        self.data_dir = conf.get_string('data_dir')

        with open(os.path.join(self.data_dir, "train.txt"), "r") as f:
            json_files = f.read().split("\n")
        self.json_files = [os.path.join(self.data_dir, json_p) for json_p in json_files]

        all_rgbs, all_rays, all_sun_dirs, all_ids = [], [], [], []
        for t, json_p in enumerate(self.json_files):
            with open(json_p) as f:
                d = json.load(f)
            img_p = os.path.join(self.data_dir, d["img"])
            img_id = os.path.splitext(os.path.basename(d["img"]))[0]

            rgbs = sat_utils.load_tensor_from_rgb_geotiff(img_p, self.img_downscale)

            cache_path = "{}/{}.data".format(self.data_dir, img_id)
            if os.path.exists(cache_path) and self.conf.get_bool('force_reload'):
                rays = torch.load(cache_path)
            else:
                h, w = int(d["height"] // self.img_downscale), int(d["width"] // self.img_downscale)
                rpc = sat_utils.rescale_rpc(rpcm.RPCModel(d["rpc"], dict_format="rpcm"), 1.0 / self.img_downscale)
                min_alt, max_alt = float(d["min_alt"]), float(d["max_alt"])
                cols, rows = np.meshgrid(np.arange(w), np.arange(h))
                rays = self.get_rays(cols.flatten(), rows.flatten(), rpc, min_alt, max_alt)
                torch.save(rays, cache_path)
            sun_dirs = sat_utils.get_sun_dirs(float(d["sun_elevation"]), float(d["sun_azimuth"]), rays.shape[0])

            all_ids += [t * torch.ones(rays.shape[0], 1)]
            all_rgbs += [rgbs]
            all_rays += [rays]
            all_sun_dirs += [sun_dirs]
            
            print("Image {} loaded ( {} / {} )".format(img_id, t + 1, len(json_files)))

        self.all_ids = torch.cat(all_ids, 0).cpu()
        self.all_rays = torch.cat(all_rays, 0)  # (len(json_files)*h*w, 8)
        self.all_rgbs = torch.cat(all_rgbs, 0)  # (len(json_files)*h*w, 3)
        self.all_sun_dirs = torch.cat(all_sun_dirs, 0)  # (len(json_files)*h*w, 3)
        pts3d_utm_dir = os.path.join(self.data_dir, "ba_files/ba_params/utm_pts3d.npy")
        self.pts3d_utm = torch.from_numpy(np.load(pts3d_utm_dir)).float().cuda()
        self.cal_scaling_params(self.all_rays)
        self.all_rays = self.normalize_rays(self.all_rays)
        self.all_rays = torch.hstack([self.all_rays, self.all_sun_dirs])  # (len(json_files)*h*w, 11)
        self.pts3d_utm = self.normalize_pts3d(self.pts3d_utm)

        self.image_shapes = []
        self.offsets = []
        self.all_depth_map = []
        self.all_normalized_depth_map = []
        self.load_and_flatten_depth_maps()

        self.val_range = None
        with open(os.path.join(self.data_dir, "{}_DSM.txt".format(img_id[:7])), "r") as f:
            val_data = f.read().split("\n")
        self.resolution = int(float(val_data[2]))
        self.val_range = torch.tensor([[float(val_data[0]), float(val_data[1])],
                                       [float(val_data[0]) + float(val_data[2]) * float(val_data[3]), float(val_data[1]) + float(val_data[2]) * float(val_data[3])]]).cpu()
        self.val_range[:, 0] -= self.center[0]
        self.val_range[:, 1] -= self.center[1]
        self.val_range = torch.cat((self.val_range, torch.tensor([float(d["min_alt"]) - self.center[2], float(d["max_alt"]) - self.center[2]]).unsqueeze(-1).cpu()), -1)

        self.val_range[:, 0] /= self.range
        self.val_range[:, 1] /= self.range
        self.val_range[:, 2] /= self.range

        print('Load data: End')

    def cal_scaling_params(self, all_rays):
        near_points = all_rays[:, :3]
        far_points = all_rays[:, :3] + all_rays[:, 7:8] * all_rays[:, 3:6]
        all_points = torch.cat([near_points, far_points], 0)

        d = {}
        d["X_scale"], d["X_offset"] = sat_utils.rpc_scaling_params(all_points[:, 0])
        d["Y_scale"], d["Y_offset"] = sat_utils.rpc_scaling_params(all_points[:, 1])
        d["Z_scale"], d["Z_offset"] = sat_utils.rpc_scaling_alt_params(all_points[:, 2])
        self.center = torch.tensor([float(d["X_offset"]), float(d["Y_offset"]), float(d["Z_offset"])]).cpu()
        self.range = torch.max(torch.tensor([float(d["X_scale"]), float(d["Y_scale"]), float(d["Z_scale"])])).cpu()

    def normalize_rays(self, rays):
        rays[:, 0] -= self.center[0]
        rays[:, 1] -= self.center[1]
        rays[:, 2] -= self.center[2]
        rays[:, 0] /= self.range
        rays[:, 1] /= self.range
        rays[:, 2] /= self.range
        rays[:, 6] /= self.range
        rays[:, 7] /= self.range
        return rays

    def normalize_pts3d(self, pts3d):
        pts3d[:, 0] -= self.center[0]
        pts3d[:, 1] -= self.center[1]
        pts3d[:, 2] -= self.center[2]

        pts3d[:, 0] /= self.range
        pts3d[:, 1] /= self.range
        pts3d[:, 2] /= self.range
        return pts3d

    def load_and_flatten_depth_maps(self):
        txt_file = self.data_dir + "/train_depth.txt"
        flattened_depths = []
        normalized_depths = []

        with open(txt_file, 'r') as f:
            filenames = f.readlines()

        for filename in filenames:
            path = filename.strip()
            path = self.data_dir + path
            if not os.path.isfile(path):
                print(f"Warning: file not found {path}")
                continue

            depth_img = Image.open(path)
            depth_array = np.array(depth_img).astype(np.float32)[:, :, 0]  # shape [H, W]
            self.image_shapes += [depth_array.shape]
            depth_flat = torch.from_numpy(depth_array).view(-1)  # shape [H*W]
            d_min = depth_flat.min()
            d_max = depth_flat.max()
            depth_normalize = (depth_flat - d_min) / (d_max - d_min + 1e-6)
            depth_normalize = 1 - depth_normalize

            flattened_depths += [depth_flat]
            normalized_depths += [depth_normalize]

        start_idx = 0
        for h, w in self.image_shapes:
            num_pixels = h * w
            self.offsets.append((start_idx, h, w))
            start_idx += num_pixels

        self.all_depth_map = torch.cat(flattened_depths, 0)
        self.all_normalized_depth_map = torch.cat(normalized_depths, 0)

    def get_rays(self, cols, rows, rpc, min_alt, max_alt):
        min_alts = float(min_alt) * np.ones(cols.shape)
        max_alts = float(max_alt) * np.ones(cols.shape)
        lons, lats = rpc.localization(cols, rows, max_alts)
        x_near, y_near, z_near = sat_utils.latlon_to_ecef_custom(lats, lons, max_alts)
        xyz_near = np.vstack([x_near, y_near, max_alts]).T

        lons, lats = rpc.localization(cols, rows, min_alts)
        x_far, y_far, z_far = sat_utils.latlon_to_ecef_custom(lats, lons, min_alts)
        xyz_far = np.vstack([x_far, y_far, min_alts]).T

        rays_o = xyz_near  

        d = xyz_far - xyz_near
        rays_d = d / np.linalg.norm(d, axis=1)[:, np.newaxis]

        fars = np.linalg.norm(d, axis=1)
        nears = float(0) * np.ones(fars.shape)

        rays = torch.from_numpy(np.hstack([rays_o, rays_d, nears[:, np.newaxis], fars[:, np.newaxis]]))
        rays = rays.type(torch.FloatTensor)
        return rays
    
    def __len__(self):
        return self.all_rays.shape[0]

    def __getitem__(self, idx):
        if self.train:
            return self.all_rays[idx], self.all_rgbs[idx], self.all_depth_map[idx], self.all_normalized_depth_map[idx]
        else:
            with open(self.json_files[idx]) as f:
                d = json.load(f)
            return (self.all_rays[torch.where(self.all_ids == idx)[0]], int(d["height"] // self.img_downscale), int(d["width"] // self.img_downscale))
