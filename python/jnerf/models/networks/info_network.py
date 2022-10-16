import jittor as jt
import jittor.nn as nn

import numpy as np
# from jnerf.models.position_encoders.neus_encoder.embedder import get_embedder
from jnerf.utils.config import get_cfg
from jnerf.utils.registry import build_from_cfg, NETWORKS, ENCODERS

# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**jt.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = jt.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return jt.concat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [jt.sin, jt.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

# Model
@NETWORKS.register_module()
class InfoNeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        super(InfoNeRF, self).__init__()
        self.D = D
        self.W = W
        cfg = get_cfg()
        output_ch = 5 if cfg.N_importance > 0 else 4
        self.pos_encoder, self.input_ch = get_embedder(cfg.multires)
        # self.input_ch = input_ch
        # self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D-1)])
                
        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
            self.dir_encoder, self.input_ch_views = get_embedder(cfg.multires_views)
            self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_views + W, W//2)])
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def execute(self, input_pts, input_views=None):
        N_samples = input_pts.shape[1]
        input_pts = self.pos_encoder(input_pts.reshape(-1, input_pts.shape[-1]))        
        h = input_pts
        # print("inputs pts: ", input_pts, input_pts.shape)
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = nn.relu(h)
            if i in self.skips:
                h = jt.concat([input_pts, h], -1)

        if self.use_viewdirs:
            input_views = self.dir_encoder(input_views[:,None,:].expand(1,N_samples,1).reshape(-1, input_views.shape[-1]))
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = jt.concat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = nn.relu(h)
            rgb = self.rgb_linear(h)
            outputs = jt.concat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)
        return outputs.reshape(-1, N_samples, outputs.shape[-1])    

@NETWORKS.register_module()
class InfoNeRF_RGB(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False, alpha_model=None):
        """ 
        """
        super(InfoNeRF_RGB, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            # self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

        self.alpha_model = alpha_model

    def execute(self, input_pts, input_views):
        # input_pts, input_views = jt.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = nn.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            with jt.no_grad():
                alpha = self.alpha_model(x)[...,3][...,None]
            feature = self.feature_linear(h)
            h = jt.concat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = nn.relu(h)
            rgb = self.rgb_linear(h)
            outputs = jt.concat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs    