from functools import wraps
from typing import Sequence, Any
import jax
from flax import nnx
from einops import rearrange
jnp = jax.numpy

def edm_precond(model_fn, *, sigma_data=0.5):
    """https://arxiv.org/abs/2206.00364"""
    @wraps(model_fn)
    def _f(x, sigma, *args, x_cond=None, **kwargs):
        sigma = jnp.asarray(sigma, dtype=x.dtype)

        if sigma.size == 1:
            sigma = jnp.repeat(sigma, x.shape[0])
        if sigma.ndim == 1:
            assert len(sigma) == x.shape[0], f"Size mismatch between sigma: {sigma.shape} and x_t {x.shape}."
            sigma = sigma.reshape((-1,) + ((1,) * (x.ndim - 1)))

        c_skip = sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)
        c_out = jnp.sqrt(sigma * sigma_data / (sigma ** 2 + sigma_data ** 2))
        c_in = 1 / jnp.sqrt(sigma ** 2 + sigma_data ** 2)

        c_noise = jnp.log(sigma.reshape(-1)) / 4 

        x_in = c_in * x
        if x_cond is not None:
            x_in = jnp.concatenate([x_in, x_cond], axis=-1)

        F_x = model_fn(x_in, c_noise, *args, **kwargs)

        D_x = c_skip * x + c_out * F_x

        return D_x

    return _f


class LabelEmbedding(nnx.Module):
    def __init__(self, emb_dim, label_dim=-1, endpoint=False, max_positions=10000, *, dtype=None, rngs=nnx.Rngs(0)):
        freqs = jnp.arange(emb_dim // 2)
        freqs = freqs / (emb_dim // 2 - (1 if endpoint else 0))
        freqs = (1 / max_positions) ** freqs
        self.freqs = freqs
        self.noise_proj_1 = nnx.Linear(emb_dim, emb_dim, dtype=dtype, rngs=rngs)
        self.noise_proj_2 = nnx.Linear(emb_dim, emb_dim, dtype=dtype, rngs=rngs)

        if label_dim > 0:
            self.label_proj = nnx.Linear(
                label_dim, emb_dim, use_bias=False,
                kernel_init=nnx.initializers.zeros, dtype=dtype, rngs=rngs)

    def __call__(self, noise_label, class_labels=None):
        x = jnp.asarray(noise_label).reshape(-1, 1) * self.freqs
        x = jnp.concatenate([jnp.cos(x), jnp.sin(x)], axis=-1)

        x = self.noise_proj_1(x)
        x = jax.nn.silu(x)
        x = self.noise_proj_2(x)

        if class_labels is not None:
            label_emb = self.label_proj(class_labels)
            x += label_emb
        x = jax.nn.silu(x)

        return x


class DMConvBlock(nnx.Module):
    def __init__(self, dim, emb_dim, drop_rate=0., kernel_size=7, use_att=False, *, dtype=None, rngs=nnx.Rngs(0)):
        dargs = dict(dtype=dtype, rngs=rngs)
        self.conv = nnx.Conv(
            dim, dim, (kernel_size, kernel_size), feature_group_count=dim, use_bias=False, **dargs)
        
        self.norm = nnx.LayerNorm(dim, use_scale=False, **dargs)

        self.mlp_in = nnx.Linear(dim, dim * 4, **dargs)
        self.mlp_out = nnx.Linear(dim * 4, dim, use_bias=False, **dargs)
        
        self.emb_norm = nnx.LayerNorm(dim, use_bias=False, use_scale=False, rngs=rngs)
        self.emb_proj = nnx.Linear(emb_dim, dim * 2, **dargs)

        if use_att:
            self.att = nnx.MultiHeadAttention(dim//64, dim, decode=False, **dargs)
        else:
            self.att = None

        self.dropout = nnx.Dropout(drop_rate, broadcast_dims=(-1,-2,-3), rngs=rngs)

    def __call__(self, x: jnp.ndarray, emb: jnp.ndarray):
        shortcut = x

        x = self.conv(x)
        x = self.norm(x)
        x = self.mlp_in(x)
        x = nnx.gelu(x)
        x = self.mlp_out(x)

        emb = rearrange(emb, 'b c -> b 1 1 c')
        scale, shift = jnp.split(self.emb_proj(emb), 2, -1)
        x = self.emb_norm(x) * scale + shift

        if self.att is not None:
            b, h, w, c = x.shape
            x = self.att(x.reshape(b, h*w, c))
            x = x.reshape(b, h, w, c)

        x = self.dropout(x)

        x = x + shortcut

        return x


class UNetDown(nnx.Module):
    def __init__(self, *, 
            in_dim: int, 
            patch_size: int, 
            model_dim: int, 
            emb_dim: int, 
            depths: Sequence[int], 
            dim_multi: Sequence[int],
            att_levels: Sequence[int]=[],
            cond_dim: int=-1,
            dropout_rates: float=0., 
            rngs:nnx.Rngs=nnx.Rngs(0), 
            dtype:Any=None, 
    ):
        dargs = dict(dtype=dtype, rngs=rngs)

        if not isinstance(dropout_rates, Sequence):
            dropout_rates = [dropout_rates] * len(depths)

        self.layers = []
        for k in range(len(depths)):
            module = nnx.Module()
            dim = model_dim * dim_multi[k]
            if k == 0:
                module.down = nnx.Conv(
                    in_dim, dim, (patch_size, patch_size), (patch_size, patch_size), **dargs)
                module.norm = nnx.LayerNorm(dim, rngs=rngs)
            else:
                prev_dim = model_dim * dim_multi[k-1]
                module.down = nnx.Conv(prev_dim, dim, (2, 2), (2, 2), **dargs)
                module.norm = nnx.LayerNorm(prev_dim, rngs=rngs)

            module.blocks = [
                DMConvBlock(dim, emb_dim, drop_rate=dropout_rates[k], use_att=k in att_levels,**dargs) 
                for _ in range(depths[k])]

            if cond_dim > 0:
                module.merge = nnx.Linear(cond_dim * dim_multi[k], dim, **dargs)

            self.layers.append(module)


    def __call__(self, x, emb, conds=None):
        skips = []
        for k, module in enumerate(self.layers):
            if k == 0:
                x = module.norm(module.down(x))
            else:
                x = module.down(module.norm(x))

            if conds is not None:
                x += module.merge(conds[k])

            skips.append(x)

            for block in module.blocks:
                x = block(x, emb)
                skips.append(x)
        
        return skips


class UNetUp(nnx.Module):
    def __init__(self, *,
            out_dim: int, 
            patch_size: int, 
            model_dim: int, 
            emb_dim: int, 
            depths: Sequence[int], 
            dim_multi: Sequence[int],
            att_levels: Sequence[int]=[],
            cond_dim: int=-1,
            dropout_rates:float=0., 
            rngs:nnx.Rngs=nnx.Rngs(0), 
            dtype:Any=None,
    ):
        dargs = dict(dtype=dtype, rngs=rngs)
        if not isinstance(dropout_rates, Sequence):
            dropout_rates = [dropout_rates] * len(depths)

        self.layers = []
        for k in range(len(depths)):
            module = nnx.Module()
            dim = model_dim * dim_multi[k]
            if k == 0:
                module.up = nnx.ConvTranspose(
                    dim, out_dim, (patch_size, patch_size), (patch_size, patch_size), **dargs)
            else:
                module.up = nnx.ConvTranspose(dim, model_dim * dim_multi[k-1], (2, 2), (2, 2), **dargs)
            module.norm = nnx.LayerNorm(dim, rngs=rngs)
            module.blocks = [
                DMConvBlock(dim, emb_dim, drop_rate=dropout_rates[k], use_att=k in att_levels, **dargs) 
                for _ in range(depths[k])]

            if cond_dim > 0:
                module.merge = nnx.Linear(cond_dim * dim_multi[k], dim, **dargs)

            self.layers.append(module)

        # ks = max(patch_size // 2, 3)
        # self.final_conv = nnx.ConvTranspose(
        #     model_dim * dim_multi[0], out_dim, (ks, ks), (patch_size//2, patch_size//2), **dargs)

    def __call__(self, skips, emb, conds=None):
        x = 0.
        n_lvls = len(self.layers)
        for k, module in enumerate(self.layers[::-1]):
            if conds is not None:
                x += module.merge(conds[n_lvls-k-1])

            for block in module.blocks:
                x = block(x + skips.pop(), emb)
 
            x = module.up(module.norm(x + skips.pop()))
        
        assert len(skips) == 0
        
        # X = self.final_conv(nnx.gelu(x))

        return x

