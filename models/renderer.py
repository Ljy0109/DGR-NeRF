import torch
import torch.nn.functional as F
import numpy as np
import mcubes


def extract_fields(bound_min, bound_max, resolution, query_func):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return u, vertices, triangles


def sample_pdf(bins, weights, n_samples, det=False):
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples])

    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


class NeuSRenderer:
    def __init__(self,
                 sdf_network,
                 deviation_network,
                 color_network,
                 n_samples,
                 n_importance,
                 up_sample_steps,
                 perturb):
        self.sdf_network = sdf_network
        self.deviation_network = deviation_network
        self.color_network = color_network
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.up_sample_steps = up_sample_steps
        self.perturb = perturb

    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        prev_cos_val = torch.cat([torch.zeros([batch_size, 1]), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)   
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, last=False):

        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape

        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)

        if not last:
            new_sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf
    
    def render_core(self,
                    rays_o,
                    rays_d,
                    sun_dir,
                    z_vals,
                    sample_dist,
                    sdf_network,
                    deviation_network,
                    color_network,
                    cos_anneal_ratio=0.0):
        batch_size, n_samples = z_vals.shape

        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, sample_dist], -1)
        mid_z_vals = z_vals + dists * 0.5

        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
        dirs = rays_d[:, None, :].expand(pts.shape)
        sun_dirs = sun_dir[:, None, :].expand(pts.shape)

        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)
        sun_dirs = sun_dirs.reshape(-1, 3)

        sdf_nn_output = sdf_network(pts)
        sdf = sdf_nn_output[:, :1]
        feature_vector = sdf_nn_output[:, 1:]

        gradients = sdf_network.gradient(pts).squeeze()
        sampled_color = color_network(pts, gradients, dirs, sun_dirs, feature_vector)
        sampled_color = (sampled_color).reshape(batch_size, n_samples, 3)

        inv_s = deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)           # Single parameter
        inv_s = inv_s.expand(batch_size * n_samples, 1)

        true_cos = (dirs * gradients).sum(-1, keepdim=True)

        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)

        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)

        transparency = torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        weights = alpha * transparency
        color = (sampled_color * weights[:, :, None]).sum(dim=1)

        depth_z = (mid_z_vals[..., :, None] * weights[:, :, None]).sum(dim=1)
        with torch.no_grad():
            z_diff = depth_z - mid_z_vals - 0.05  
            weights_z = torch.exp(-10 * z_diff) 

            mask = z_diff > 0  
            weights_z = weights_z * mask  

        loss_outside = (F.relu(-sdf.view(batch_size, -1)) * weights_z).sum() / (weights_z.sum() + 1e-6)

        gradient_error = torch.mean((torch.linalg.norm(gradients, ord=2,dim=-1) - 1.0) ** 2)

        return {
            'color': color,
            'sdf': sdf,
            'gradients': gradients.reshape(batch_size, n_samples, 3),
            's_val': 1.0 / inv_s,
            'weights': weights,
            'cdf': c.reshape(batch_size, n_samples),
            'gradient_error': gradient_error,
            'pts': pts.reshape(batch_size, n_samples, 3),
            'depth_z': depth_z,
            'loss_outside': loss_outside
        }

    def render(self, rays_o, rays_d, near, far, sun_dir, perturb_overwrite=-1, cos_anneal_ratio=0.0):

        batch_size = len(rays_o)    
        sample_dist = far / self.n_samples   
        z_vals = torch.linspace(0.0, 1.0, self.n_samples)   
        z_vals = near + (far - near) * z_vals[None, :]  

        n_samples = self.n_samples
        perturb = self.perturb

        if perturb > 0:
            t_rand = (torch.rand([batch_size, 1]) - 0.5)
            z_vals = z_vals + t_rand * far.min() / self.n_samples

        if self.n_importance > 0:
            with torch.no_grad():   
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
              
                sdf = self.sdf_network.sdf(pts.reshape(-1, 3).cuda()).reshape(batch_size, self.n_samples)

                for i in range(self.up_sample_steps):
                    new_z_vals = self.up_sample(rays_o,
                                                rays_d,
                                                z_vals,
                                                sdf,
                                                self.n_importance // self.up_sample_steps,
                                                64 * 2**i)
                    z_vals, sdf = self.cat_z_vals(rays_o,
                                                  rays_d,
                                                  z_vals,
                                                  new_z_vals,
                                                  sdf,
                                                  last=(i + 1 == self.up_sample_steps))
            n_samples = self.n_samples + self.n_importance

        ret_fine = self.render_core(rays_o,
                                    rays_d,
                                    sun_dir,
                                    z_vals,
                                    sample_dist,
                                    self.sdf_network,
                                    self.deviation_network,
                                    self.color_network,                                    
                                    cos_anneal_ratio=cos_anneal_ratio)

        return {
            'color_fine': ret_fine['color'],
            's_val': ret_fine['s_val'].reshape(batch_size, n_samples).mean(),
            'cdf_fine': ret_fine['cdf'],
            'weight_max': torch.max(ret_fine['weights'], dim=-1, keepdim=True)[0],
            'gradients': ret_fine['gradients'],
            'weights': ret_fine['weights'],
            'gradient_error': ret_fine['gradient_error'],
            'pts': ret_fine['pts'],
            'depth_z': ret_fine['depth_z'],
            'loss_outside': ret_fine['loss_outside']
            # 'single_intersection_loss': single_intersection_loss,
        }

    def single_z_intersection_sdf_loss(self, rays_o, val_range, sdf_network, num_samples=64, sample_ratio=0.25):
        """
        rays_o: (B, 3) 光线起点
        val_range: (2, 3), 表示 AABB 的最小最大值 (xyz)
        sdf_network: 神经网络模型，输入 3D 点 (N, 3)，输出 sdf 值 (N,)
        num_samples: 每条 ray 上采样的点数
        """
        device = rays_o.device
        B = rays_o.shape[0]
        num_sampled_rays = int(B * sample_ratio)
        rand_idx = torch.randperm(B, device=device)[:num_sampled_rays]
        rays_o = rays_o[rand_idx]

        rays_d = torch.tensor([0, 0, 1], device=device).unsqueeze(0).repeat(num_sampled_rays, 1)  # (B', 3)

        near = 0.0
        far = (val_range[1, 2] - val_range[0, 2]).item()

        z_vals = torch.linspace(near, far, num_samples, device=device).unsqueeze(0).repeat(num_sampled_rays, 1)  # (B', N)

        points = rays_o.unsqueeze(1) + z_vals.unsqueeze(2) * rays_d.unsqueeze(1)  # (B', N, 3)

        sdf_nn_output = sdf_network(points.view(-1, 3))
        sdf_vals = sdf_nn_output[:, :1].view(num_sampled_rays, num_samples)
        sign = (sdf_vals >= 0).float()  # (B, N)
        sign_change = torch.abs(sign[:, 1:] - sign[:, :-1])  # (B', N-1)
        flip_count = torch.sum(sign_change, dim=1)  # (B',)

        penalty = torch.clamp(flip_count - 1, min=0.0)  

        return penalty.mean()

    def extract_geometry(self, bound_min, bound_max, resolution, threshold=0.0):
        return extract_geometry(bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=lambda pts: -self.sdf_network.sdf(pts))
