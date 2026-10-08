import os
import torch
import trimesh
import logging
import argparse
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
from shutil import copyfile
from pyhocon import ConfigFactory
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.dataset import Dataset
from models.renderer import NeuSRenderer
from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork
import torchsort


class Runner:
    def __init__(self, conf_path, mode='train', case='CASE_NAME', is_continue=False):
        self.device = torch.device('cuda')

        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.dataset = Dataset(self.conf['dataset'])
        self.data_loader = DataLoader(self.dataset,
                                      shuffle=True,
                                      num_workers=1,  # 1,
                                      batch_size=self.conf['train.batch_size'], # self.conf['train.batch_size']
                                      pin_memory=True,
                                      generator=torch.Generator(device='cuda'))
        self.data_loader_iter = iter(self.data_loader)
        self.iter_step = 0

        self.end_iter = self.conf.get_int('train.end_iter')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.lr = None
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)

        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None
        self.epoch = 1

        params_to_train = []
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())

        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

        self.renderer = NeuSRenderer(self.sdf_network,
                                     self.deviation_network,
                                     self.color_network,
                                     **self.conf['model.neus_renderer'], )

        latest_model_name = None
        if is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

        if self.mode[:5] == 'train':
            self.file_backup()

    def train(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step
        self.pts3d_utm = self.dataset.pts3d_utm
        for iter_i in tqdm(range(res_step)):
            try:
                rays, true_rgb, depth_map, normalized_depth_map = next(self.data_loader_iter)
            except StopIteration:
                self.epoch += 1
                self.data_loader_iter = iter(self.data_loader)
                rays, true_rgb, depth_map, normalized_depth_map = next(self.data_loader_iter)

            if self.epoch <= 10:
                stage = 2
            elif self.epoch <= 20:
                stage = 3
            elif self.epoch <= 30:
                stage = 4
            else:
                stage = 4
            self.sdf_network.set_pose_enc_freq(stage)

            rays = rays.to(self.device)
            true_rgb = true_rgb.to(self.device)

            rays_o, rays_d, near, far, sun_dir = rays[:, :3], rays[:, 3: 6], rays[:, 6: 7], rays[:, 7: 8], rays[:,
                                                                                                           8: 11]
            render_out = self.renderer.render(rays_o, rays_d, near, far, sun_dir,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio())

            color_fine = render_out['color_fine']
            s_val = render_out['s_val']
            cdf_fine = render_out['cdf_fine']
            gradient_error = render_out['gradient_error']
            weight_max = render_out['weight_max']
            depth_z = render_out['depth_z']
            loss_outside = render_out['loss_outside']

            # Loss
            color_error = color_fine - true_rgb
            color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='mean')
            psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb) ** 2).sum() / (color_fine.shape[0] * 3.0)).sqrt())

            eikonal_loss = gradient_error

            loss_sdf = self.gaussian_sdf_loss(torch.abs(self.sdf_network.sdf(self.pts3d_utm)))
            loss_range = self.range_penalty_loss(depth_z)

            if self.iter_step > 4000:
                loss_depth_z = self.soft_rank_loss(normalized_depth_map, depth_z)
                loss = color_fine_loss + eikonal_loss * self.igr_weight + loss_depth_z * 0.01 + loss_sdf * 0.05 + (loss_range + loss_outside) * 1.0
            else:
                loss = color_fine_loss + eikonal_loss * self.igr_weight + loss_sdf * 0.05 + (loss_range + loss_outside) * 1.0

            print('total_loss = ', loss)

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

            self.iter_step += 1

            self.writer.add_scalar('Loss/loss', loss, self.iter_step)
            self.writer.add_scalar('Loss/color_loss', color_fine_loss, self.iter_step)
            self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)
            self.writer.add_scalar('Statistics/lr', self.lr, self.iter_step)
            self.writer.add_scalar('Statistics/s_val', s_val, self.iter_step)
            self.writer.add_scalar('Statistics/cdf', (cdf_fine[:, :1]).mean(), self.iter_step)
            self.writer.add_scalar('Statistics/weight_max', (weight_max).mean(), self.iter_step)
            self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)
            self.writer.add_scalar('Statistics/epoch', self.epoch, self.iter_step)

            self.update_learning_rate()
            if self.iter_step != 0 and self.iter_step % 10000 == 0:
                self.save_checkpoint()
                self.validate_image()
                self.validate_mesh(resolution=self.dataset.resolution)


    def range_penalty_loss(self, t, t_min=0.05, t_max=0.99, penalty_weight=1.0):
        over_max = torch.relu(t - t_max)
        under_min = torch.relu(t_min - t)

        loss = penalty_weight * (over_max + under_min).mean()
        return loss


    def soft_rank_loss(self, depth_map, depth_z, tolerance=0.5):
        depth_map = depth_map.to(depth_z.device)

        if depth_z.dim() == 2:
            depth_z = depth_z.squeeze(-1)

        depth_map = (depth_map - depth_map.mean()) / (depth_map.std() + 1e-8)
        depth_z = (depth_z - depth_z.mean()) / (depth_z.std() + 1e-8)

        depth_map = depth_map.unsqueeze(0)
        depth_z = depth_z.unsqueeze(0)

        rank_map = torchsort.soft_rank(depth_map, regularization='l2')
        rank_z = torchsort.soft_rank(depth_z, regularization='l2')

        rank_diff = (rank_map - rank_z).abs()
        penalized_diff = F.relu(rank_diff - tolerance)
        loss = (penalized_diff ** 2).mean()
        return loss


    def gaussian_sdf_loss(self, abs_sdf, delta=0.05):
        # if self.iter_step < 1000:
        #     delta = 0.05
        # elif self.iter_step >= 1000 or self.iter_step < 2000:
        #     delta = 0.02
        # else:
        #     delta = 0.01
        return (1 - torch.exp(-(abs_sdf / delta) ** 2)).mean()

    def relu_sdf_loss(self, abs_sdf, delta=0.02):
        return (F.relu(abs_sdf - delta) ** 2).mean()

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha
            # self.depth_wight = np.sin(np.pi * progress * 0.5)

        self.lr = self.learning_rate * learning_factor
        for g in self.optimizer.param_groups:
            g['lr'] = self.lr

    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))
        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name),
                                map_location=self.device, weights_only=False)
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']
        self.epoch = checkpoint['epoch']

        logging.info('End')

    def save_checkpoint(self):
        checkpoint = {
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
            'epoch': self.epoch
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint,
                   os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def validate_image(self, idx=-1, resolution_level=-1):
        if idx < 0:
            idx = np.random.randint(len(self.dataset.json_files))
        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))

        self.dataset.train = False
        rays, H, W = self.dataset[idx]
        self.dataset.train = True
        rays_o, rays_d, near, far, sun_dirs = rays[:, :3].cuda(), rays[:, 3:6].cuda(), rays[:, 6:7].cuda(), rays[:,
                                                                                                            7:8].cuda(), rays[
                                                                                                                         :,
                                                                                                                         8:].cuda()
        rays_o = rays_o.split(self.batch_size)
        rays_d = rays_d.split(self.batch_size)
        near = near.split(self.batch_size)
        far = far.split(self.batch_size)
        sun_dirs = sun_dirs.split(self.batch_size)

        out_rgb_fine = []
        out_normal_fine = []
        for rays_o_batch, rays_d_batch, near_batch, far_batch, sun_dirs_batch in zip(rays_o, rays_d, near, far,
                                                                                     sun_dirs):
            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near_batch,
                                              far_batch,
                                              sun_dirs_batch,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio())

            def feasible(key):
                return (key in render_out) and (render_out[key] is not None)

            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            if feasible('gradients') and feasible('weights'):
                n_samples = self.renderer.n_samples + self.renderer.n_importance  
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None] 
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)
            del render_out

        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = (np.concatenate(out_normal_fine, axis=0).reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)

        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)

        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'validations_fine',
                                        '{:0>8d}_{}_{}_rgb.png'.format(self.iter_step, i, idx)),
                           img_fine[..., i][:, :, [2, 1, 0]])
            if len(out_normal_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'normals',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           normal_img[..., i])


    def validate_mesh(self, resolution=512, threshold=0.0):
        u, vertices, triangles = self.renderer.extract_geometry(self.dataset.val_range[0], self.dataset.val_range[1],
                                                                resolution=resolution + 1, threshold=threshold)
        vertices *= self.dataset.range.numpy()
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)))

        u = np.flip(u, axis=2)
        sdf_0 = np.zeros((resolution + 1, resolution + 1), dtype=np.float32)
        dsm = np.zeros((resolution, resolution), dtype=np.float32)
        for i in range(u.shape[0]):
            for j in range(u.shape[1]):
                for k in range(u.shape[2]):
                    if u[i, j, k] >= 0.0:
                        pre = u[i, j, k - 1]
                        post = u[i, j, k]
                        sdf_0[i, j] = k + (pre / (pre - post))
                        break
        for i in range(resolution):
            for j in range(resolution):
                dsm[i, j] = (sdf_0[i, j] + sdf_0[i + 1, j] + sdf_0[i, j + 1] + sdf_0[i + 1, j + 1]) * 0.25
        dsm = dsm / resolution
        dsm = self.dataset.val_range[1, 2].numpy() - dsm * (
                    self.dataset.val_range[1, 2].numpy() - self.dataset.val_range[0, 2].numpy())
        dsm = dsm * self.dataset.range.numpy() + self.dataset.center[2].numpy()
        dsm = np.rot90(dsm, 1)
        cv.imwrite(os.path.join(self.base_exp_dir, 'validations_fine', '{:0>8d}_dsm.tif'.format(self.iter_step)), dsm)

        logging.info('End')


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FORMAT = "%(message)s"
    logging.basicConfig(level=logging.INFO, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='/root/mycode/DGR/config.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='JAX_068')

    args = parser.parse_args()
    # args.case = 'JAX_207'

    torch.cuda.set_device(args.gpu)
    runner = Runner(args.conf, args.mode, args.case, args.is_continue)

    if args.mode == 'train':
        runner.train()
    elif args.mode == 'validate_image':
        runner.validate_image()
    elif args.mode == 'validate_mesh':
        runner.validate_mesh(resolution=852, threshold=args.mcube_threshold)
