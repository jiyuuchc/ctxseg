import math
import jax
import numpy as np
from functools import partial
from typing import Callable, Sequence
from flax import nnx
from jax.typing import ArrayLike

import logging
logger = logging.getLogger(__name__)

jnp = jax.numpy


def edm_sigma_steps(sigma_min, sigma_max, num_steps, *, rho=7):
    sigma_steps = np.linspace(
        sigma_max ** (1 / rho), 
        sigma_min ** (1 / rho), 
        num_steps, 
    ) ** rho

    return sigma_steps


def vp_sigma_steps(sigma_min, sigma_max, num_steps, *, epsilon_s = 1e-3):
    vp_beta_d = 2 * (np.log(sigma_min ** 2 + 1) / epsilon_s - np.log(sigma_max ** 2 + 1)) / (epsilon_s - 1)
    vp_beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d
    vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
    orig_t_steps = np.linspace(1, epsilon_s, num_steps)
    sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)

    return sigma_steps


def ve_sigma_steps(sigma_min, sigma_max, num_steps):
    step_indices = np.linspace(0, 1, num_steps)
    sigma_steps = (sigma_min / sigma_max) ** step_indices * sigma_max

    return sigma_steps


def iddpm_sigma_steps(sigma_min, sigma_max, num_steps, *, C_1=0.001, C_2=0.008, M=1000):
    step_indices = np.linspace(0, 1, num_steps)
    u = np.zeros(M + 1)
    alpha_bar = lambda j: np.sin(0.5 * np.pi * j / M / (C_2 + 1)) ** 2
    for j in np.arange(M, 0, -1): # M, ..., 1
        u[j - 1] = np.sqrt((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1)
    u_filtered = u[(u >= sigma_min) & (u <= sigma_max)]
    sigma_steps = u_filtered[((len(u_filtered) - 1) / (num_steps - 1) * step_indices).astype(int)]

    return sigma_steps


def dps_gradient(
    model_fn:Callable,
    tangents_fn:Callable|None=None,
    *,
    dps_weight:float=1.0,
    inpaint_weight:float=8.0,
):
    def _f(
        x_t, 
        sigma,     
        inpaint:ArrayLike|None=None, 
        inpaint_mask: ArrayLike|None=None,     
        **kwargs
    ):
        assert x_t.ndim == 4

        model_fn_ = partial(model_fn, sigma=sigma, **kwargs)

        denoised, vjp_f = jax.vjp(model_fn_, x_t)

        if tangents_fn is not None:
            tangents = tangents_fn(denoised, sigma) * dps_weight
        else:
            tangents = jnp.zeros_like(x_t)

        if inpaint_mask is not None:
            tangents = jnp.where(inpaint_mask, (inpaint - denoised) * inpaint_weight, tangents)

        guidance = vjp_f(tangents)[0]

        return (x_t - denoised) / sigma - guidance

    return _f


def dm_gradient(
    model_fn:Callable, 
    *, 
    beta:float = 0.0,
):
    def _f(
        x_t, 
        sigma,     
        inpaint:ArrayLike|None=None, 
        inpaint_mask: ArrayLike|None=None,     
        **kwargs
    ):
        w1 = 1 / sigma
        w2 = w1 + beta
        Dx = model_fn(x_t, sigma, **kwargs)
        if inpaint_mask is not None:
            Dx = jnp.where(inpaint_mask, inpaint, Dx)
        g = x_t * w1 - Dx * w2
        return g

    return _f


def seq_gradient(
    model_fn:Callable,
    *,
    beta:float = 0.0,
    weight:float = 0.1,
    inpaint_weight:float=8.0,
):
    def _f(
        x_t, 
        sigma,     
        inpaint:ArrayLike|None=None, 
        inpaint_mask: ArrayLike|None=None,     
        **kwargs
    ):
        @partial(jax.value_and_grad, has_aux=True)
        def _mc(x_t):
            e_x0 = model_fn(x_t, sigma, **kwargs)

            cost = 0.5 * weight * ((e_x0[1:] - e_x0[:-1]) ** 2).sum()

            if inpaint_mask is not None:
                cost += 0.5 * inpaint_weight * ((e_x0 - inpaint) ** 2).sum(where=inpaint_mask)

            return cost, e_x0

        w1 = 1 / sigma
        w2 = w1 + beta

        (cost, e_x0), g = _mc(x_t)

        return x_t * w1 - e_x0 * w2 + g

    return _f


def sde_churn(
    x_cur, 
    sigma_cur, 
    *,
    gamma:float=math.sqrt(2)-1,
    S_min:float=0, 
    S_max:float=float('inf'), 
    S_noise:float=1,
 ):
    ''' Temperarily increase noise
        gamma: scale of noise churning
        S_min: sigma range for churning, min value
        S_max: sigma range for churning, max value
        S_noise=1: noise scaling during churning
    '''
    if S_min <= sigma_cur <= S_max and gamma > 0:
        sigma_hat = sigma_cur * (1 +  gamma)
        e = np.random.normal(size=x_cur.shape, dtype=x_cur.dtype)
        x_hat = x_cur + S_noise * math.sqrt(sigma_hat ** 2 - sigma_cur ** 2) * e
    else:
        sigma_hat, x_hat = sigma_cur, x_cur

    return x_hat, sigma_hat


def _get_discretization_fn(discretization):
    # Time step discretization.

    if discretization == "edm":
        discretization_fn = edm_sigma_steps
    elif discretization == "ve":
        discretization_fn = ve_sigma_steps
    elif discretization == "iddpm":
        discretization_fn = iddpm_sigma_steps
    elif discretization == 'vp':
        discretization_fn = vp_sigma_steps
    else:
        assert isinstance(discretization, Callable), f"Unknown discretization fn {discretization}"
        discretization_fn = discretization

    return discretization_fn


def _get_churn_fn(churn):
    if churn == "none":
        churn_fn = lambda x, sigma: (x, sigma)
    elif churn == "sde":
        churn_fn = sde_churn
    else:
        assert isinstance(churn, Callable), f"unknown churn fn {churn}"
        churn_fn = churn

    return churn_fn


def dm_sampler(
    gradient_fn,
    latent_shape:Sequence[int]|ArrayLike,
    *,
    num_samples:int=1,
    num_steps:int=18, 
    sigma_min:float=0.002, 
    sigma_max:float=80,
    solver:str='heun', 
    discretization:str|Callable='edm',
    churn:str|Callable="none",
    resample_fn:Callable|None=None,
    rngs: nnx.Rngs|None=None,
    **kwargs,
)->jax.Array|list[jax.Array]:
    """    
    DM sampler with simplified parameterization and generalization.

    Args:
        gradient_fn: f(x_t, sigma, **kwargs) supply gradient for the update step. Typically obtained 
            by calling: dm_gradient(model_fn)
        latent_shape: Shape of the latent, e.g. (batch_size, 512, 512, 2)
    Keyward Args:
        num_samples: how many samples to draw.
        num_steps: number of integration steps 
        sigma_min: minimal sigma discretization
        sigma_max: maxmimal sigma discretization
        solver: 'heun'|'euler'|'edf', SDE solver.
            euler: simple solver by DDIM. x_{t-1} = x_t + dsigma * S(. ; t)
            edf: 2nd order solver. x_{t-1} = x_t + dsigma * [2 * S(. ; t) - S(. ; t+1)] 
                see https://arxiv.org/abs/2306.04848
            heun: 2nd order solver. x_{t-1} = x_t + dsigma / 2 * [S(. ; t) + S_hat(. ; t-1)], where S_hat
                is the gradient estimated at the projected x_{t-1} of euler solver. See https://arxiv.org/abs/2206.00364
        discretization: 'edm'|'vp'|'ve'|'iddpm'|custum discreization function
        churn: 'none'|'sde'|custom churn function
        resample_fn: Callback function: f(g_t:List[Array], x_t:List[Array], sigma:float) which 
            can modify current intermediates including g_t: current gradients, x_t: current latents.
            g_t and x_t are lists whose lengths are num_samples
        rngs: RNG sequence. Use the "sampler" key.
        **kwargs: additional argument will be passed to model

    Returns: Sample or a list of samples if num_samples > 1 
    """
    assert solver in ['euler', 'heun', 'edf']

    if rngs is None:
        rngs = nnx.Rngs(np.random.randint(0, 1000000))

    discretization_fn = _get_discretization_fn(discretization)
    sigma_steps = discretization_fn(sigma_min, sigma_max, num_steps)
    sigma_steps = np.r_[sigma_steps, 0] # t_N = 0

    churn_fn = _get_churn_fn(churn)

    if isinstance(latent_shape, ArrayLike):
        x_next = list[latent_shape] if num_samples > 1 else [latent_shape]
        latent_shape = latent_shape[0].shape
    else:
        x_next = [
            jax.random.normal(key, latent_shape) * sigma_steps[0]
            for key in jax.random.split(rngs.sampler(), num_samples)
        ]

    # Main sampling loop.
    d_cur = None
    for sigma_cur, sigma_next in zip(sigma_steps[:-1], sigma_steps[1:]):
        # draw samples
        logger.debug(f"Sampler: compute gradient at sigma = {sigma_cur:.2f}")

        x_cur = x_next

        d_prev, d_cur = d_cur, []
        for i in range(num_samples):
            x_cur[i], sigma_hat = churn_fn(x_cur[i], sigma_cur)
            d_cur.append(gradient_fn(x_cur[i], sigma_hat, **kwargs))
        
        if resample_fn is not None:
            resample_fn(d_cur, x_cur, sigma_hat, rngs=rngs)

        # Apply 2nd order correction.
        if solver == "edf" and d_prev:
            d_cur = [-dp + 2 * d for dp, d in zip(d_prev, d_cur)]

        elif solver == "heun" and sigma_next > 0:
            logger.debug(f"seq_sampler: perform 2nd order correction")
            for i in range(num_samples):
                xn = x_cur[i] + (sigma_next - sigma_hat) * d_cur[i]
                d_prime = gradient_fn(xn, sigma_next, **kwargs)
                d_cur[i] = (d_cur[i] + d_prime) / 2
        
        x_next = [x + (sigma_next - sigma_hat) * d for x, d in zip(x_cur, d_cur)]

    if num_samples == 1:
        x_next = x_next[0]

    return x_next

