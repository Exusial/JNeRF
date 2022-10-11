import jittor as jt
from jittor import nn
import numpy as np
import os

from jnerf.utils.config import get_cfg
from jnerf.utils.registry import SAMPLERS

# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / jt.sum(weights, -1, keepdim=True)
    cdf = jt.cumsum(pdf, -1)
    cdf = jt.cat([jt.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = jt.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = jt.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbersd
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = jt.array(u)

    # Invert CDF
    # TODO: support CUDA??
    inds = jt.searchsorted(cdf, u, right=True)
    below = jt.max(jt.zeros_like(inds-1), inds-1)
    above = jt.min((cdf.shape[-1]-1) * jt.ones_like(inds), inds)
    inds_g = jt.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = jt.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = jt.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = jt.where(denom<1e-5, jt.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples

def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False, 
               out_alpha=False, out_sigma=False, out_dist=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-jt.exp(-act_fn(raw)*dists)
    
    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = jt.cat([dists, jt.array([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * jt.norm(rays_d[...,None,:], dim=-1)

    rgb = jt.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = jt.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = jt.array(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    sigma = F.relu(raw[...,3]+noise)
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * jt.cumprod(jt.cat([jt.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = jt.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = jt.sum(weights * z_vals, -1)
    disp_map = 1./jt.max(1e-10 * jt.ones_like(depth_map), depth_map / jt.sum(weights, -1))
    acc_map = jt.sum(weights, -1)
    
    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])
    
    others = {}
    if out_alpha or out_sigma or out_dist:
        if out_alpha:
            others['alpha'] = alpha
        if out_sigma:
            others['sigma'] = sigma
        if out_dist:
            others['dists'] = dists
        return rgb_map, disp_map, acc_map, weights, depth_map, others
    return rgb_map, disp_map, acc_map, weights, depth_map


def sample_sigma(rays_o, rays_d, viewdirs, network, z_vals):
    # N_rays = rays_o.shape[0]
    # N_samples = len(z_vals)
    # z_vals = z_vals.expand([N_rays, N_samples])

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
    raw = network(pts, viewdirs, network)

    rgb = jt.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    sigma = F.relu(raw[...,3])

    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d)

    return rgb, sigma, depth_map


def visualize_sigma(sigma, z_vals, filename):
    plt.plot(z_vals, sigma)
    plt.xlabel('z_vals')
    plt.ylabel('sigma')
    plt.savefig(filename)
    return

@SAMPLERS.register_module()
class InfoSampler():
    def __init__(self):
        cfg = get_cfg()
        self.using_fp16 = cfg.using_fp16
        self.near = cfg.near
        self.far = cfg.far
        pass
    
    def sample_uniform(self, rays_o, rays_d, N_samples, perturb, lindisp=False, pytest=False):
        near, far = self.near, self.far
        N_rays = rays_o.shape[0]
        t_vals = jt.linspace(0., 1., steps=N_samples)
        if not lindisp:
            z_vals = near * (1.-t_vals) + far * (t_vals)
        else:
            z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

        z_vals = z_vals.expand([N_rays, N_samples])

        if perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            upper = jt.concat([mids, z_vals[...,-1:]], -1)
            lower = jt.concat([z_vals[...,:1], mids], -1)
            # stratified samples in those intervals
            t_rand = jt.rand(z_vals.shape)

            # Pytest, overwrite u with numpy's fixed random numbers
            if pytest:
                np.random.seed(0)
                t_rand = np.random.rand(*list(z_vals.shape))
                t_rand = jt.array(t_rand)

            z_vals = lower + (upper - lower) * t_rand

        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
        return pts, z_vals

    def sample_fine(self, rays_o, rays_d, z_vals, weights, N_importance, perturb=0., pytest=False):
        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = jt.argsort(jt.cat([z_vals, z_samples], -1), -1)[0]
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]
        return pts, z_vals

    def rays2rgb(self, network_outputs, training_background_color, inference=False):
        if self.using_fp16:
            with jt.flag_scope(auto_mixed_precision_level=5):
                return self.rays2rgb_(network_outputs, training_background_color, inference)
        else:
            return self.rays2rgb_(network_outputs, training_background_color, inference)
    
    def raw2outputs(self, raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False, 
               out_alpha=False, out_sigma=False, out_dist=False):
        return raw2outputs(raw, z_vals, rays_d, raw_noise_std=raw_noise_std, white_bkgd=white_bkgd, pytest=pytest, 
               out_alpha=out_alpha, out_sigma=out_sigma, out_dist=out_dist)

    def rays2rgb_(self,):
        pass