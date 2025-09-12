from typing import Sequence
import jax
from flax import nnx
from jax.typing import ArrayLike
jnp = jax.numpy

class GRN(nnx.Module):
    def __init__(self, dim, *, epsilon:float=1e-6):
        self.epsilon = epsilon
        self.beta = nnx.Param(jnp.zeros([1,1,1,dim]))
        self.gamma = nnx.Param(jnp.zeros([1,1,1,dim]))

    def __call__(self, x):
        mu2 = jax.lax.square(jnp.abs(x)).mean(axis=(-2, -3), keepdims=True)
        mu2 = jnp.maximum(mu2, 1e-6)
        Gx = jax.lax.sqrt(mu2)
        Nx = Gx / (Gx.mean(axis=-1, keepdims=True) + self.epsilon)

        return (self.gamma * (x * Nx) + self.beta + x).astype(x.dtype)   


class ConvBlock(nnx.Module):
    """ConvNeXt Block.
    """
    def __init__(self, input_dim, dim, drop_rate=0.2, kernel_size=7, dtype=None, rngs=nnx.Rngs(0)):
        dargs = dict(dtype=dtype, rngs=rngs)
        self.conv = nnx.Conv(
            input_dim, dim, 
            kernel_size=(kernel_size, kernel_size),
            feature_group_count=dim,
            **dargs)

        self.norm = nnx.LayerNorm(dim, **dargs)
        self.proj_out = nnx.Linear(dim, dim * 4, **dargs)
        self.proj_in = nnx.Linear(dim * 4, dim, **dargs)
        self.grn = GRN(dim * 4)
        self.dropout = nnx.Dropout(drop_rate, broadcast_dims=(-1,-2,-3), rngs=rngs)

    def __call__(self, x):
        shortcut = x

        x = self.conv(x)
        x = self.norm(x)
        x = self.proj_out(x)
        x = nnx.gelu(x)
        x = self.grn(x)
        x = self.proj_in(x)
        x = self.dropout(x)

        x = x + shortcut

        return x


class ConvDecoder(nnx.Module):
    def __init__(self, out_dim, patch_size, model_dim, depths, dim_multi, skip_multi=None, rngs=nnx.Rngs(0), dtype=None):
        dargs = dict(dtype=dtype, rngs=rngs)

        self.layers = []
        for k in range(len(depths)):
            module = nnx.Module()
            dim = model_dim * dim_multi[k]
            if k == 0:
                module.up = nnx.ConvTranspose(
                    dim, out_dim, 
                    (patch_size, patch_size), (patch_size, patch_size), **dargs)
            else:
                module.up = nnx.ConvTranspose(
                    dim, model_dim * dim_multi[k-1], (2, 2), (2, 2), **dargs)

            module.norm = nnx.LayerNorm(dim, rngs=rngs)

            module.blocks = [ConvBlock(dim, dim, **dargs,) 
                for _ in range(depths[k])]
            
            if skip_multi is not None:
                module.skip_proj = nnx.Linear(skip_multi[k] * model_dim, dim, **dargs)
            else:
                module.skip_proj = lambda x: x

            self.layers.append(module)


    def __call__(self, skips):
        x = 0.
        for module in self.layers[::-1]:
            x = x + module.skip_proj(skips.pop())
            for block in module.blocks:
                x = block(x) 
            
            x = module.up(module.norm(x))

        assert len(skips) == 0

        return x



class ConvNeXt(nnx.Module):
    def __init__(self, 
        in_dim:int=3,
        patch_size:int=4,
        depths:Sequence[int]=(3, 3, 27, 3),
        model_dim:int=128,
        dim_multi:Sequence[int]=(1,2,4,8),
        drop_rate:float=0.,
        *, decoder=None, rngs=nnx.Rngs(0), dtype=None,
    ):
        dargs = dict(dtype=dtype, rngs=rngs)

        dp_rate = 0.
        self.layers = []
        for k in range(len(depths)):
            module = nnx.Module()
            if k == 0:
                module.down = nnx.Conv(
                    in_dim, model_dim * dim_multi[0], 
                    kernel_size=(patch_size, patch_size),
                    strides=(patch_size, patch_size),
                    **dargs)
                module.norm = nnx.LayerNorm(model_dim * dim_multi[0], rngs=rngs)
            else:
                module.down = nnx.Conv(
                    model_dim * dim_multi[k-1],
                    model_dim * dim_multi[k],
                    kernel_size=(2, 2),
                    strides=(2, 2),
                    **dargs)
                module.norm = nnx.LayerNorm(model_dim * dim_multi[k-1], rngs=rngs)

            module.blocks = [ConvBlock(
                model_dim * dim_multi[k],
                model_dim * dim_multi[k],
                drop_rate=dp_rate, **dargs,
            ) for _ in range(depths[k])]
    
            dp_rate += drop_rate / (len(depths) - 1)

            self.layers.append(module)
        
        self.decoder = decoder


    def __call__(self, x:ArrayLike)->list:
        skips = []
        for k, module in enumerate(self.layers):
            if k == 0:
                x = module.norm(module.down(x))
            else:
                x = module.down(module.norm(x))

            for b in module.blocks:
                x = b(x)

            skips.append(x)
        
        if self.decoder is not None:
            skips = self.decoder(skips)
        
        return skips
