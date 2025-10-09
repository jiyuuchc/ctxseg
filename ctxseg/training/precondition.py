from typing import Callable
import jax
import numpy as np
from flax import nnx
from ..modeling.diffusion import edm_precond

jnp = jax.numpy

def edm_schedule(key, x, *, sigma_data=0.5, sigma_min=0.002, sigma_max=80, p_mean=-1.2, p_std=1.2):
    """ Modeled after the EDM schedule as described in https://arxiv.org/abs/2006.11239
    """
    shape = x.shape[:1] + (1,) * (x.ndim - 1)
    log_sigma = jax.random.normal(key, shape) * p_std + p_mean

    sigma = jnp.clip(
        jnp.exp(log_sigma),
        sigma_min,
        sigma_max,
    )

    loss_weight = 1 / (sigma ** 2) + 1 / (sigma_data ** 2) 

    return sigma, loss_weight


def edm_hat_schedule(key, x, *, sigma_data=0.5, sigma_min=0.002, sigma_max=80, p_mean=-1.2, p_std=1.2):
    """ Modified EDM schedule with increased weights at low SNRs
    """
    from functools import lru_cache

    @lru_cache
    def _cache(sigma_data, sigma_min, sigma_max, p_mean, p_std):
        snr_min, snr_max = 1 / (sigma_max ** 2), 1 / (sigma_min ** 2)
        snrs = jnp.geomspace(snr_min, snr_max, 10000)
        p_snrs = jnp.exp(-((-jnp.log(snrs) - p_mean * 2) ** 2) / (8 * p_std * p_std)) /  snrs
        effective_weights = p_snrs * (sigma_data * sigma_data * snrs + 1)
        max_weights = jnp.max(effective_weights)
        max_snr = snrs[jnp.argmax(effective_weights)]
        
        corrections = jnp.where(
            snrs < max_snr,
            max_weights / effective_weights,
            jnp.ones_like(snrs)
        ) 
        return snrs, corrections

    bins, corrections = _cache(sigma_data, sigma_min, sigma_max, p_mean, p_std) 

    shape = x.shape[:1] + (1,) * (x.ndim - 1)
    log_sigma = jax.random.normal(key, shape) * p_std + p_mean

    sigma_hat = jnp.clip(
        jnp.exp(log_sigma),
        sigma_min,
        sigma_max,
    )
    snrs = 1 / (sigma_hat ** 2)

    loss_weight = snrs * (sigma_data ** 2) + 1
    loss_weight *= corrections[jnp.digitize(snrs, bins)]

    return sigma_hat, loss_weight


def ncsn2_schedule(key, x, *, sigma_min=0.01, sigma_max=80):
    """ Modeled after the NCSN2 schedule as described in https://arxiv.org/abs/2006.09011
    This is an ELBO schedule due to effective constant loss weight. It also uses the low
    discrenpancy sampling described in https://arxiv.org/abs/2107.00630.
    """
    B = x.shape[0]

    gamma_min = 2 * np.log(sigma_min)
    gamma_max = 2 * np.log(sigma_max)
    t = (jax.random.uniform(key, [1]) + jnp.arange(B)/B) % 1
    t = t.reshape(x.shape[:1] + (1,) * (x.ndim - 1))
    gamma = gamma_min + t * (gamma_max - gamma_min)
    sigma = jnp.exp(gamma / 2)
    loss_weight = 1 / (sigma * sigma)

    return sigma, loss_weight


def Precond(cls=None, *, schedule_fn="edm", kind="edm", **cond_kwargs):
    """ Use this as a class decorator. Creates a preconditioned diffusion model with a 
    specific training schedule. Also provids a loss method, which returns weighted L2 loss

    Keyword Args:
        schedule_fn: string or callable. Training schedule.
        **cond_kwargs: additional keyword args passed to the edm conditioner
    """
    schedule_fn_dict = {
        "edm": edm_schedule,
        "edmhat": edm_hat_schedule,
        "ncsn2": ncsn2_schedule,
    }

    if cls is None:
        return lambda cls: Precond(cls, schedule_fn=schedule_fn, kind=kind, **cond_kwargs)

    assert issubclass(cls, nnx.Module)

    if schedule_fn in schedule_fn_dict:
        schedule_fn = schedule_fn_dict[schedule_fn]

    assert isinstance(schedule_fn, Callable), f"invalid {schedule_fn=}"
    assert kind in ["edm"]

    class _class(cls):
        def __init__(self, *args, rngs, **kwargs):
            super(_class, self).__init__(*args, rngs=rngs, **kwargs)
            self.rngs = rngs

        def __call__(self, x_t, sigma, **kwargs):
            return edm_precond(super(_class, self).__call__, **cond_kwargs)(
                x_t, sigma, **kwargs
            )

        def loss(self, x_gt, **kwargs):
            sigma, loss_weight = schedule_fn(self.rngs.train(), x_gt)
            x_t = x_gt + jax.random.normal(self.rngs.train(), x_gt.shape) * sigma
            E_x0 = self(x_t, sigma, **kwargs)
            loss = 0.5 * (E_x0 - x_gt) ** 2
            loss = loss * loss_weight

            return loss.mean()

    return _class
