import pickle
import random
from pathlib import Path

import jax
import numpy as np
import tifffile
from absl import app, flags
from flax import nnx
from tqdm import tqdm

from ctxseg.sampling.samplers import dm_sampler
from ctxseg.modeling.diffusion import edm_precond
from ctxseg.segmentation.utils import center_crop, pad_channel

flags.DEFINE_string("model", None, "", required=True)
flags.DEFINE_string("datapath", None, "", required=True)
flags.DEFINE_string("dataset", None, "",)
flags.DEFINE_integer("ps", 4, "")
flags.DEFINE_boolean("crop", True, "whether crop (deafult) or resize the image")
flags.DEFINE_string("solver", "edf", "dm solver")
flags.DEFINE_string("discretization", "edm", "dm discretization")
flags.DEFINE_integer("nsteps", 1, "dm steps")
flags.DEFINE_integer("nsamples", 8, "number of samples for each image")
flags.DEFINE_string("logpath", "dm_prediction", "dir to save inference results")
flags.DEFINE_float("sigmamax", 80, "ODE max sigma")
flags.DEFINE_float("sigmamin", 0.002, "ODE min sigma")


jnp = jax.numpy
rngs = nnx.Rngs(random.randint(0, 10000))

FLAGS = flags.FLAGS

@nnx.jit
def model_fn(model, x_t, sigma):
    return edm_precond(model.predict, sigma_data=1.0)(x_t, sigma)


def predict(model, image, n, **kwargs):
    model.set_image(image[None])
    if FLAGS.nsteps > 1:
        samples = dm_sampler(
            lambda x_t, sigma: (x_t - model_fn(model, x_t, sigma))/sigma,
            (n, 512, 512, 2),
            sigma_max=FLAGS.sigmamax,
            sigma_min=FLAGS.sigmamin,
            solver=FLAGS.solver, 
            num_steps=FLAGS.nsteps, 
            discretization=FLAGS.discretization,
            rngs=rngs,
            **kwargs,
        )
    else:
        sigma = 80.
        x_t = jax.random.normal(rngs.default(), (n, 512, 512, 2))
        samples = model_fn(model, x_t * sigma, sigma)

    return samples


def segment_image(model, image, gt_mask):
    n = FLAGS.nsamples
    image = pad_channel(image/image.max())
    h, w = image.shape[:2]

    if FLAGS.crop:
        image = center_crop(image)
        gt_mask = center_crop(gt_mask).squeeze(-1)

        gt_mask = gt_mask[:h, :w]

        flow = predict(model, image, n)
        flow = flow[:, :h, :w]

        image = image[:h, :w]
    
    else: # resize
        img = jax.image.resize(image, (512, 512, 3), 'linear')
        flow = predict(model, img, n)
        flow = jax.image.resize(flow, (n, h, w, 2), 'linear')

    return flow, image, gt_mask


def load_model():
    from ctxseg.training.utils import EMAOptimizer
    cp = Path(FLAGS.model)

    with open(cp, "rb") as f:
        target = pickle.load(f)

    if isinstance(target, EMAOptimizer):
        model = target.ema_model
    elif isinstance(target, nnx.Optimizer):
        model = target.model
    elif isinstance(target, nnx.Module):
        model = target
    else:
        from ctxseg.modeling.ctxseg import CtxSegP
    
        model = CtxSegP(patch_size=FLAGS.ps, rngs=rngs)
        nnx.update(model, target)
        model.set_ref()

    del target

    return model


def get_ds(name):
    ds_path = Path(FLAGS.datapath) / name
    if ds_path.exists() and ds_path.is_dir():
        def _ds_gen():
            import imageio.v2 as imageio
            for mask_fn in (ds_path).glob("*_label.tif"):
                mask = imageio.imread(mask_fn).astype("uint16")
                img_name = mask_fn.name.replace("_label","")
                img = imageio.imread(ds_path/img_name)
                yield dict(
                    image=img.astype('float32'),
                    label=mask.astype('uint16'),
                    img_id=img_name.split('.')[0],
                )
        return _ds_gen()
    else:
        raise ValueError(f"dataset {name} not found")


def main(_):
    model = load_model()

    if FLAGS.dataset is None:
        p = Path(FLAGS.datapath)
        datasets = [x.name for x in p.iterdir() if x.is_dir()]
    else:
        datasets = [FLAGS.dataset]

    for ds_name in datasets:
        print(f"============ {ds_name} ==========")
        ds = get_ds(ds_name)
        dst_dir = Path(FLAGS.logpath)/ds_name
        dst_dir.mkdir(exist_ok=True, parents=True)
        for data in tqdm(ds):
            flow, img, gt = segment_image(model, data['image'], data['label'])
            tifffile.imwrite(dst_dir/f'{data["img_id"]}_pred.tif', np.moveaxis(flow, -1, 0))
            tifffile.imwrite(dst_dir/f'{data["img_id"]}.tif', (img*255).astype('uint8'))
            tifffile.imwrite(dst_dir/f'{data["img_id"]}_label.tif', gt.astype('uint16'))

if __name__ == "__main__":
    app.run(main)
