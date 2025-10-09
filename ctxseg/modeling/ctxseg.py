import jax
from einops import rearrange, repeat
from flax import nnx
from .convnext import ConvNeXt, ConvDecoder
from .diffusion import UNetDown, UNetUp, LabelEmbedding
jnp = jax.numpy

def sigmoid_att(q, k, v, mask, temp):
    attn = jnp.einsum("...qd,...kd->...qk", q,k) / temp
    if mask is not None:
        attn = jax.nn.sigmoid(attn) * mask
        attn = attn /  (1e-6 + jnp.sum(mask, axis=-1, keepdims=True))
    else:
        attn = jax.nn.sigmoid(attn)
        attn = attn / attn.shape[-1]
    x = jnp.einsum("...qk,...kd->...qd", attn, v) 

    return x

def softmax_att(q, k, v, mask, temp):
    attn = jnp.einsum("...qd,...kd->...qk", q,k) / temp
    if mask is not None:
        attn = jnp.where(mask, attn, -1e6)
    attn = jax.nn.softmax(attn)
    x = jnp.einsum("...qk,...kd->...qd", attn, v)
    return x

def linear_attn(q, k, v, mask, temp):
    if mask is not None:
        mask = jnp.swapaxes(mask, -1, -2)
        k = jnp.where(mask, k, k.min())
        v = jnp.where(mask, v, 0.)

    q = jax.nn.softmax(q, axis=-1) / temp
    k = jax.nn.softmax(k, axis=-2)

    context = jnp.einsum('bhnd,bhne->bhde', k, v)

    x = jnp.einsum('bhnd,bhde->bhne', q, context)

    return x


class CtxAtt(nnx.Module):
    def __init__(self, num_heads, in_dim, k_dim=None, v_dim=None, hidden_dim=None, use_bias=True, att_type="sigmoid", *, rngs=nnx.Rngs(0), dtype=None):
        dargs = dict(dtype=dtype, rngs=rngs)
        hidden_dim = hidden_dim or in_dim
        assert hidden_dim % num_heads == 0, f"hidden_dim {hidden_dim} should be divisible by num_heads {num_heads}"

        self.proj_q = nnx.Linear(in_dim, hidden_dim, use_bias=use_bias, **dargs)
        self.proj_k = nnx.Linear(k_dim or in_dim, hidden_dim, use_bias=use_bias, **dargs)
        self.proj_v = nnx.Linear(v_dim or in_dim, hidden_dim, use_bias=use_bias, **dargs)
        self.proj_out = nnx.Linear(hidden_dim, hidden_dim, **dargs)

        self.num_heads = num_heads
        self.temp = (hidden_dim // num_heads) ** .5
        if att_type == "sigmoid":
            self.att_fn = sigmoid_att
        elif att_type == "softmax":
            self.att_fn = softmax_att
        else:
            assert att_type == "linear"
            self.att_fn = linear_attn

    def __call__(self, q, k=None, v=None, mask=None):
        B, R, C = q.shape[:3]
        if mask is not None:
            mask = jax.image.resize(mask, (B, R, C), method='nearest')
            mask = rearrange(mask, 'B R C -> B 1 1 (R C)')

        shortcut = q

        k = q if k is None else k
        v = q if v is None else v
        q = rearrange(self.proj_q(q), 'B R C (H D) -> B H (R C) D', H=self.num_heads)
        k = rearrange(self.proj_k(k), 'B R C (H D) -> B H (R C) D', H=self.num_heads)
        v = rearrange(self.proj_v(v), 'B R C (H D) -> B H (R C) D', H=self.num_heads)

        x = self.att_fn(q, k, v, mask, self.temp)

        x = rearrange(x, "B H (R C) D -> B R C (H D)", R=R, C=C) 

        x = self.proj_out(x) + shortcut

        return x

    def batch_process(self, q, k, v, mask=None):
        n_refs = k.shape[0]
        B = q.shape[0]
        q = repeat(q, 'B ... -> (B N) ...', N=n_refs)
        k = repeat(k, 'N ... -> (B N) ...', B=B)
        v = repeat(v, 'N ... -> (B N) ...', B=B)
        if mask is not None:
            mask = repeat(mask, 'N ... -> (B N) ...', B=B)

        x = self(q, k, v, mask)

        x = rearrange(x, '(B N) ... -> B N ...', B=B).mean(axis=1)

        return x 


class CtxSegP(nnx.Module):
    def __init__(self, model_dim=128, patch_size=4, depths=(2,2,2,2), dim_multi=(1,2,4,8), *, rngs=nnx.Rngs(0),  dtype=None):
        self.label_emb = LabelEmbedding(model_dim, endpoint=True, rngs=rngs, dtype=dtype)
        self.x_encoder = ConvNeXt(model_dim=model_dim, patch_size=patch_size, rngs=rngs, dtype=dtype)
        self.y_encoder = UNetDown(
            in_dim=2, patch_size=patch_size, model_dim=model_dim, emb_dim=model_dim,
            depths=depths, dim_multi=dim_multi, att_levels=(2,3),
            rngs=rngs, dtype=dtype)
        self.decoder = UNetUp(
            out_dim=2, patch_size=patch_size, model_dim=model_dim, emb_dim=model_dim,
            depths=depths, dim_multi=dim_multi, att_levels=(2,3), cond_dim=model_dim,
            rngs=rngs, dtype=dtype)

        self.ref_emb = nnx.Param(jnp.zeros((1, 128)))
        self.atts = []
        for k, multi in enumerate(dim_multi):
            dim = model_dim * multi
            self.atts.append([
                CtxAtt(dim//128, dim, att_type="linear" if k==0 else "sigmoid", rngs=rngs, dtype=dtype)
                for _ in range(depths[k] + 1)
            ])

        self.set_ref()

    def __call__(self, x_t, noise_labels, *, image, image_ref=None, flow_ref=None, ref_mask=None):
        emb = self.label_emb(noise_labels)

        if image_ref is None:
            x_skips = self.x_encoder(image)
            y_skips = self.y_encoder(x_t, emb)

        else:
            x_skips = self.x_encoder(image)
            y_skips = self._ctx_encode_y(
                x_t, emb,
                x_skips, 
                self.x_encoder(image_ref),
                self.y_encoder(flow_ref, jnp.broadcast_to(self.ref_emb, emb.shape)),
                ref_mask,
            )

        out = self.decoder(y_skips, emb, conds=x_skips)

        return out


    def _ctx_encode_y(self, y, emb, x_skips, x_ref, y_ref, ref_mask):
        y_skips = []
        for k, module in enumerate(self.y_encoder.layers):
            mask = jax.image.resize(ref_mask, x_skips[k].shape[:3], method='nearest')

            if k == 0:
                y = module.norm(module.down(y))
            else:
                y = module.down(module.norm(y))
            
            y += self.atts[k][0](x_skips[k], x_ref[k], y_ref.pop(0), mask)

            y_skips.append(y)

            for m, block in enumerate(module.blocks):
                y = block(y, emb)
                y += self.atts[k][m+1](x_skips[k], x_ref[k], y_ref.pop(0), mask)
                y_skips.append(y)
        
        return y_skips


    @nnx.jit
    def set_ref(self, ref_image=None, ref_flow=None, ref_mask=None):
        if ref_image is None:
            self.x_ref = self.y_refs = None
        else:
            self.x_ref = self.x_encoder(ref_image)
            self.y_ref = self.y_encoder(ref_flow, jnp.repeat(self.ref_emb, ref_image.shape[0], 0))
        
        self.ref_mask = ref_mask
    

    @nnx.jit
    def set_image(self, image):
        x_skips = self.x_encoder(image)
        self.x_skips = x_skips
        
        if self.x_ref is not None:
            x_ref, y_ref = self.x_ref, self.y_ref.copy()
            self.ref_features = []
            for k, module in enumerate(self.y_encoder.layers):
                if self.ref_mask is not None:
                    mask = jax.image.resize(self.ref_mask, x_ref[k].shape[:3], method='nearest')
                else:
                    mask = jnp.ones(x_ref[k].shape[:3], dtype=bool)

                mask = repeat(mask, 'N ... -> (B N) ...', B=image.shape[0])
                x_ref_ = repeat(x_ref[k], 'N ... -> (B N) ...', B=image.shape[0])
                x_ = repeat(x_skips[k], 'B ... -> (B N) ...', N=x_ref[k].shape[0])

                for att in self.atts[k]:
                    y_ref_ = repeat(y_ref.pop(0), 'N ... -> (B N) ...', B=image.shape[0])
                    yy = att(x_, x_ref_, y_ref_, mask)
                    yy = rearrange(yy, '(B N) ... -> B N ...', B=image.shape[0])
                    self.ref_features.append(yy.mean(axis=1))

    @nnx.jit
    def predict(self, x_t, noise_labels):
        emb = self.label_emb(noise_labels)
        x_skips = self.x_skips

        if self.x_ref is None:
            y_skips = self.y_encoder(x_t, emb)
        else:
            y = x_t
            ref_features = self.ref_features.copy()
            y_skips = []
            for k, module in enumerate(self.y_encoder.layers):
                if k == 0:
                    y = module.norm(module.down(y))
                else:
                    y = module.down(module.norm(y))
                
                y += ref_features.pop(0)

                y_skips.append(y)

                for m, block in enumerate(module.blocks):
                    y = block(y, emb)
                    y += ref_features.pop(0)
                    y_skips.append(y)

        out = self.decoder(y_skips, emb, conds=x_skips)

        return out


class CtxSegD(nnx.Module):
    def __init__(self, *, ps=4, rngs=nnx.Rngs(0), dtype=None):
        self.encoder = ConvNeXt(patch_size=ps, rngs=rngs, dtype=dtype)
        self.ref_encoder = ConvNeXt(patch_size=ps, model_dim=64, in_dim=2, rngs=rngs, dtype=dtype)
        self.decoder = ConvDecoder(3, patch_size=ps, model_dim=128, depths=(2,2,2,1), dim_multi=(4,4,4,8), skip_multi=(1,2,4,8), rngs=rngs, dtype=dtype)
        self.atts = []
        for k in range(4):
            q_dim = 128 * (8 if k==3 else 4)
            kv_dim = 192 * (2 ** k)
            self.atts.append(CtxAtt(q_dim // 128, q_dim, kv_dim, kv_dim, att_type="linear" if k<2 else "sigmoid", rngs=rngs))


    def __call__(self, image, ref_image=None, ref_flow=None, ref_mask=None):
        skips = self.encoder(image)

        if ref_image is not None:
            assert ref_flow is not None
            x_ref = self.encoder(ref_image)
            y_ref = self.ref_encoder(ref_flow)

        x = 0.
        for k in reversed(range(len(skips))):
            module = self.decoder.layers[k]

            x = x + module.skip_proj(skips.pop())
            if ref_image is not None:
                q = jnp.c_[x_ref[k], y_ref[k]]
                x = self.atts[k](x, q, q, ref_mask)

            for block in module.blocks:
                x = block(x) 
            
            x = module.up(module.norm(x))

        assert len(skips) == 0

        return x

    @nnx.jit
    def set_ref(self, ref_image=None, ref_flow=None, ref_mask=None):
        if ref_image is None:
            self.x_ref = self.y_ref = None
        else:
            assert ref_flow is not None
            self.x_ref = self.encoder(ref_image)
            self.y_ref = self.ref_encoder(ref_flow)
        self.ref_mask = ref_mask


    @nnx.jit
    def predict(self, image):
        try:
            ref_mask = self.ref_mask
            x_ref = self.x_ref
            y_ref = self.y_ref
        except:
            ref_mask = None
            x_ref = y_ref = None

        skips = self.encoder(image)

        x = 0.
        for k in reversed(range(len(skips))):
            module = self.decoder.layers[k]
            x = x + module.skip_proj(skips.pop())
            if x_ref is not None:
                assert y_ref is not None
                q = jnp.c_[x_ref[k], y_ref[k]]
                x = self.atts[k].batch_process(x, q, q, ref_mask)

            for block in module.blocks:
                x = block(x)

            x = module.up(module.norm(x))

        assert len(skips) == 0

        return x
