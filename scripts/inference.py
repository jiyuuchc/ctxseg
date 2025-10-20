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
flags.DEFINE_string("discretization", "", "dm discretization")
flags.DEFINE_integer("nsteps", 4, "dm steps")
flags.DEFINE_integer("nsamples", 1, "number of samples for each image")
flags.DEFINE_string("logpath", "dm_prediction", "dir to save inference results")
flags.DEFINE_float("sigmamax", 80, "ODE max sigma")
flags.DEFINE_float("sigmamin", 0.002, "ODE min sigma")
flags.DEFINE_boolean("logimage", True, "whether to log images")


jnp = jax.numpy
rngs = nnx.Rngs(random.randint(0, 10000))

FLAGS = flags.FLAGS

@nnx.jit
def model_fn(model, x_t, sigma):
    return edm_precond(model.predict, sigma_data=1.0)(x_t, sigma)


def _get_schedule():
    from ctxseg.sampling.samplers import _get_discretization_fn
    disc = FLAGS.discretization
    if disc == "":
        sigma_steps = [FLAGS.sigmamax / (2 ** k) for k in range(FLAGS.nsteps)]
    else:
        discretization_fn = _get_discretization_fn(disc)
        sigma_steps = discretization_fn(FLAGS.sigmamin, FLAGS.sigmamax, FLAGS.nsteps)
    
    return sigma_steps


def _predict(model, image, n, **kwargs):
    model.set_image(image[None])
    sigmas = _get_schedule()

    sigma = sigmas[0]
    latent = jax.random.normal(rngs.sample(), (n, 512, 512, 2)) * sigma
    e_x = model_fn(model, latent, sigma)

    for sigma_next in sigmas[1:]:
        r = sigma_next / sigma
        latent = latent * r + e_x * (1 - r)
        e_x = model_fn(model, latent, sigma_next)
        sigma = sigma_next

    samples = e_x

    return samples


# def _predict(model, image, n, **kwargs):
#     model.set_image(image[None])
#     sigmas = _get_schedule()

#     n_ = n
#     sigma = sigmas[0]
#     latent = jax.random.normal(rngs.sample(), (n_, 512, 512, 2)) * sigma
#     x_t = model_fn(model, latent, sigma)
#     samples=[x_t]

#     for sigma_next in sigmas[1:]:
#         r = sigma_next / sigma
#         latent = jax.random.normal(rngs.sample(), (2, n_, 512, 512, 2)) * sigma_next
#         latent += x_t * (1-r)
#         latent = latent.reshape(-1, 512, 512, 2)
#         e_x = model_fn(model, latent, sigma_next)
#         samples.append(e_x)

#         x_t = jnp.repeat(x_t, 2, axis=0)
#         x_t = x_t * r + e_x

#         n_ *= 2
#         sigma = sigma_next

#     samples = jnp.concatenate(samples)

#     return samples

def _grpc_predict(server_url, image, n, **kwargs):
    import grpc
    import biopb.image as proto
    from biopb.image.utils import deserialize_to_numpy, serialize_from_numpy

    def _channel():
        server = server_url.split('//')[1]
        if server_url.startswith("https"):
            return grpc.secure_channel(
                target=server,
                credentials=grpc.ssl_channel_credentials(),
                options=[("grpc.max_receive_message_length", 1024 * 1024 * 512)],
            )
        else:
            return grpc.insecure_channel(
                target=server,
                options=[("grpc.max_receive_message_length", 1024 * 1024 * 512)],
            )

    with _channel() as channel:
        stub = proto.ProcessImageStub(channel)

        pixels = serialize_from_numpy(image)

        response = stub.Run(
            proto.ProcessRequest(image_data=proto.ImageData(pixels=pixels)),
            timeout=15,
        )
        output = deserialize_to_numpy(response.image_data.pixels)
        if output.shape[-1] == 1:
            output = output.squeeze(-1)

        return output


def predict(model, image, n, **kwargs):
    if isinstance(model, str):
        return _grpc_predict(model, image, n, **kwargs)
    else:
        return _predict(model, image, n, **kwargs)


def segment_image(model, image, gt_mask):
    n = FLAGS.nsamples
    image = pad_channel(image/image.max())
    h, w = image.shape[:2]

    if FLAGS.crop:
        image = center_crop(image)
        gt_mask = center_crop(gt_mask).squeeze(-1)

        gt_mask = gt_mask[:h, :w]

        results = predict(model, image, n)
        results = results[:, :h, :w]

        image = image[:h, :w]
    
    else: # resize
        from skimage.transform import resize
        img = resize(image, (512, 512))
        results = predict(model, img, n)
        results = resize(results, (results.shape[0], h, w))

    return results, image, gt_mask


def load_model():
    from ctxseg.training.utils import EMAOptimizer
    if FLAGS.model.startswith("http"):
        return FLAGS.model

    with open(FLAGS.model, "rb") as f:
        target = pickle.load(f)

    if isinstance(target, EMAOptimizer):
        model = target.ema_model
    elif isinstance(target, nnx.Optimizer):
        model = target.model
    elif isinstance(target, nnx.Module):
        model = target
    else:
        from ctxseg.modeling.diffusion import SegP
    
        model = SegP(patch_size=FLAGS.ps, rngs=rngs)
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
            results, img, gt = segment_image(model, data['image'], data['label'])

            if FLAGS.logimage:
                tifffile.imwrite(dst_dir/f'{data["img_id"]}.tif', (img*255).astype('uint8'))
                tifffile.imwrite(dst_dir/f'{data["img_id"]}_label.tif', gt.astype('uint16'))

            if isinstance(model, str):
                tifffile.imwrite(dst_dir/f'{data["img_id"]}_output.tif', results)
            else:
                tifffile.imwrite(dst_dir/f'{data["img_id"]}_pred.tif', np.moveaxis(results, -1, 0))


if __name__ == "__main__":
    app.run(main)
