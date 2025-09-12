""" Jax implementation of flow and inverse-flow operator originally used by cellpose
"""
import jax
import numpy as np
import scipy
from ..ops.ndimage import sub_pixel_samples

jnp = jax.numpy

def _extend_centers(
    neighbors, centers, isneighbor, Ly, Lx, n_iter=200,
):
    nimg = neighbors.shape[0] // 9
    pt = jnp.asarray(neighbors)

    T = jnp.zeros((nimg, Ly, Lx), dtype="float64")
    meds = jnp.asarray(centers)
    isneigh = jnp.asarray(isneighbor)

    def _inner(_T, _):
        _T = _T.at[:, meds[:, 0], meds[:, 1]].set(1e24)
        Tneigh = _T[:, pt[:, :, 0], pt[:, :, 1]]
        Tneigh *= isneigh
        _T = _T.at[:, pt[0, :, 0], pt[0, :, 1]].set(Tneigh.mean(axis=1))
        return _T, None
    
    T, _ = jax.lax.scan(_inner, T, length=n_iter)    

    T = jnp.log(1.0 + T)
    
    # gradient positions
    grads = T[:, pt[[2, 1, 4, 3], :, 0], pt[[2, 1, 4, 3], :, 1]]
    dy = grads[:, 0] - grads[:, 1]
    dx = grads[:, 2] - grads[:, 3]
    
    mu = np.stack((dy.squeeze(), dx.squeeze()), axis=-2)

    return mu


def _mask_to_flow(masks):
    if masks.max() == 0 or (masks != 0).sum() == 1:
        return np.zeros((2, *masks.shape), "float32")

    Ly0, Lx0 = masks.shape
    Ly, Lx = Ly0 + 2, Lx0 + 2

    masks_padded = np.zeros((Ly, Lx), dtype=int)
    masks_padded[1:-1, 1:-1] = masks

    # get mask pixel neighbors
    y, x = np.nonzero(masks_padded)
    neighborsY = np.stack((y, y - 1, y + 1, y, y, y - 1, y - 1, y + 1, y + 1), axis=0)
    neighborsX = np.stack((x, x, x, x - 1, x + 1, x - 1, x + 1, x - 1, x + 1), axis=0)
    neighbors = np.stack((neighborsY, neighborsX), axis=-1)

    # get mask centers
    slices = scipy.ndimage.find_objects(masks)
    assert slices is not None, f"{masks}"
    assert len(slices) > 0, f"{masks}"
    centers = np.zeros((masks.max(), 2), dtype=int)
    for i, si in enumerate(slices):
        if si is not None:
            sr, sc = si
            # ly, lx = sr.stop - sr.start + 1, sc.stop - sc.start + 1
            yi, xi = np.nonzero(masks[sr, sc] == (i + 1))
            yi = yi.astype(np.int32) + 1  # add padding
            xi = xi.astype(np.int32) + 1  # add padding
            ymed = np.median(yi)
            xmed = np.median(xi)
            imin = np.argmin((xi - xmed) ** 2 + (yi - ymed) ** 2)
            xmed = xi[imin]
            ymed = yi[imin]
            centers[i, 0] = ymed + sr.start
            centers[i, 1] = xmed + sc.start

    # get neighbor validator (not all neighbors are in same mask)
    neighbor_masks = masks_padded[neighbors[:, :, 0], neighbors[:, :, 1]]
    isneighbor = neighbor_masks == neighbor_masks[0]
    ext = np.array(
        [[s[0].stop - s[0].start + 1, s[1].stop - s[1].start + 1] for s in slices if s]
    )
    n_iter = 2 * (ext.sum(axis=1)).max()

    # run diffusion
    mu = _extend_centers(
        neighbors, centers, isneighbor, Ly, Lx, n_iter=n_iter,
    )

    # normalize
    mu /= 1e-20 + (mu ** 2).sum(axis=0) ** 0.5

    # put into original image
    mu0 = np.zeros((2, Ly0, Lx0))
    mu0[:, y - 1, x - 1] = mu

    return mu0


def mask_to_flow(mask: np.ndarray) -> np.ndarray:
    """ Convert masks to flow fields.
    Args:
        masks: 2D or 3D array of masks, where each mask is a labeled region.
    Returns:
        flows: (H, W, 2) or (D, H, W, 3) array of flow fields.
    """
    if mask.ndim == 3:
        Lz, Ly, Lx = mask.shape
        mu = np.zeros((3, Lz, Ly, Lx), np.float32)
        for z in range(Lz):
            mu0 = _mask_to_flow(mask[z])
            mu[[1, 2], z, :, :] += mu0
        for y in range(Ly):
            mu0 = _mask_to_flow(mask[:, y])
            mu[ [0, 2], :, y, :] += mu0
        for x in range(Lx):
            mu0 = _mask_to_flow(mask[:, :, x])
            mu[[0, 1], :, :, x ] += mu0

        return mu.transpose(1, 2, 3, 0)  # (D, H, W, 3)

    elif mask.ndim == 2:
        mu = _mask_to_flow(mask)

        return mu.transpose(1, 2, 0)  # (H, W, 2)

    else:
        raise ValueError("masks_to_flows only takes 2D or 3D arrays")


def follow_flows(dP, niter=200):
    """ Follow flow field to get new pixel positions.
    Args:
        dP: (H, W, 2/3) flow field.
        niter: number of iterations to follow the flow.
    Returns:
        p: (H, W, 2/3) array of new pixel positions after following the flow
    """
    d = dP.shape[-1]
    
    assert (d == 2 or d == 3) and dP.ndim == d + 1, "dP must be with shape (H, W, 2) or (D, H, W, 2)"

    if d == 2:
        H, W, _ = dP.shape
        p = np.stack(np.mgrid[:H, :W], axis=-1).astype(float)
        max_values = jnp.array([H-1, W-1])
    else:
        D, H, W, _ = dP.shape
        p = np.stack(np.mgrid[:D, :H, :W], axis=-1).astype(float)
        max_values = jnp.array([D-1, H-1, W-1])

    def _flow(_p, _):
        dPt = sub_pixel_samples(dP, _p)
        _p += jnp.clip(dPt, -1, 1)
        _p = jnp.clip(_p, 0, max_values)
        return _p, None

    p, _ = jax.lax.scan(_flow, p, length=niter)

    return p


def get_mask(p, niter=5, *, min_seed_cnts=10):
    """ Get mask from pixel positions.
    Args:
        p: (H, W, 2/3) a map of pixel positions after flow.
        niter: number of iterations to follow the flow.
    Returns:
        mask: (H, W) binary mask where pixels are inside the mask.
    """
    from scipy.ndimage import maximum_filter

    dim = p.shape[-1]

    assert (dim == 2 or dim == 3) and p.ndim == dim + 1, "p must be with shape (H, W, 2) or (D, H, W, 2)"

    p = (p + 0.5).astype(int)

    p = np.clip(p, 0, np.array(p.shape[:-1]) - 1)  # ensure p is within bounds

    assert (p >= 0).all() & (p < np.array(p.shape[:-1])).all(), "p values out of range"

    if dim == 2:
        expansion = np.stack(np.mgrid[-1:2, -1:2], axis=-1)
    else:
        expansion = np.stack(np.mgrid[-1:2, -1:2, -1:2], axis=-1)

    # get counts of postions
    p_ravel = np.ravel_multi_index(
        tuple(np.moveaxis(p, -1, 0)),
        p.shape[:-1],
    )
    p_cnts = np.bincount(p_ravel.flatten(), minlength=np.prod(p.shape[:-1]))
    p_cnts = p_cnts.reshape(p.shape[:-1])

    # get seed locations
    max_filterd_cnts = maximum_filter(p_cnts, size=5)
    seeds = np.where((p_cnts > max_filterd_cnts - 1e-6) & (p_cnts > min_seed_cnts))


    # merge with nearby 
    lut = np.zeros(p.shape[:-1], dtype=int)
    for index, seed in enumerate(np.stack(seeds, axis=-1)):
        seed_collection = seed[None, :] # start with a single seed point
        n = 1

        for _ in range(niter):
            # expand mask around the seed
            seed_collection = seed_collection + expansion.reshape(-1, 1, dim) # (8, n, 2) or (27, n, 3)
            seed_collection = seed_collection.reshape(-1, dim)
            seed_collection = seed_collection[(seed_collection > 0).all(axis=-1) & (seed_collection < p.shape[:-1]).all(axis=-1)]
            seed_cnts = p_cnts[tuple(seed_collection.T)]
            seed_collection = seed_collection[seed_cnts > 2]
            if len(seed_collection) == n:
                break
            n = len(seed_collection)

        lut[tuple(seed_collection.T)] = index + 1

    # generate mask
    mask = lut[tuple(np.moveaxis(p, -1, 0))]

    return mask


def flow_to_mask(dP, niter=200, *, min_seed_cnts=10):
    """ Convert flow field to mask.
    Args:
        dP: flow field. 
            Either batched input :(B, H, W, 2) or (B, D, H, W, 3)
            or unbatched: (H, W, 2) or (D, H, W, 3)
        niter: number of iterations to follow the flow.
    Returns:
        mask: (B, H, W) or (B, D, H, W) for batched input. (H, W) 
            or (D, H, W) for unbatched input
    """
    D = dP.shape[-1]
    assert D == 2 or D == 3, f"flow field dimsion must be 2/3, got {D}"

    if dP.ndim == D + 1: # unbatched
        p = follow_flows(dP / 5, niter=niter)

        mask = get_mask(p, min_seed_cnts=min_seed_cnts)

    else: # batched
        from functools import partial

        assert dP.ndim == D + 2, f"ilegal flow filed shape {dP.shape}"

        ps = jax.vmap(partial(follow_flows, niter=niter))(
            dP / 5
        )
        mask = np.stack([
            get_mask(p, min_seed_cnts=min_seed_cnts)
            for p in ps
        ])

    return mask

