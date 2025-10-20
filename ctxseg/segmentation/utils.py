import numpy as np

def remove_small_instances(mask, min_area=0, clean_up=True):
    """ remove all instances smaller than min_area"""
    if min_area > 0:
        unique_ids, counts = np.unique(mask, return_counts=True)
        small_ids = unique_ids[counts < min_area]

        mask = np.where(
            np.isin(mask, small_ids),
            0,
            mask,
        )

        if clean_up:
            mask = clean_up_mask(mask)
    
    return mask


def clean_up_mask(mask):
    """ ensure continuity of mask ID"""
    unique_ids = np.unique(mask)

    assert np.all(unique_ids >= 0), f"Mask containes negative values."

    lut = np.zeros(unique_ids.max()+1, dtype=mask.dtype)
    lut[unique_ids] = np.arange(len(unique_ids))

    return lut[mask]


def remove_border_instances(mask):
    """ remove all instances at image border"""
    if mask.ndim == 2:
        border_pixels = set(
            mask[0, :].tolist()
            + mask[:, -1].tolist()
            + mask[-1, :].tolist()
            + mask[:, 0].tolist()
        )
    else:
        assert mask.ndim == 3, "mask must be 2D or 3D"
        border_pixels = set(
            mask[0, :, :].reshape(-1).tolist()
            + mask[-1, :, :].reshape(-1).tolist()
            + mask[:, 0, :].reshape(-1).tolist()
            + mask[:, -1, :].reshape(-1).tolist()
            + mask[:, :, 0].reshape(-1).tolist()
            + mask[:, :, -1].reshape(-1).tolist()
        )

    if 0 in border_pixels:
        border_pixels.remove(0)
    
    mask = np.where(
        np.isin(mask, list(border_pixels)),
        0,
        mask,
    )
    
    return mask


def pad_channel(image):
    if image.ndim == 2:
        image = image[..., None]

    C = image.shape[-1]
    if C == 1:
        image = np.c_[image, image, image]
    if C == 2:
        image = np.c_[image, np.zeros_like(image[..., :1])]

    assert image.shape[-1] == 3

    return image


def center_crop(image, mask=None, *, crop_size=512):
    """ Take a (image, mask) pair. Return their center crop
    (or padding if needed) of the size (crop_size, crop_size).
    """
    if image.ndim == 2:
        image = image[...,None]

    H, W, C = image.shape
    assert mask is None or mask.shape == (H, W)

    if H < crop_size or W < crop_size:
        padding = [
            [0, max(0, crop_size-H)],
            [0, max(0, crop_size-W)],
        ]
        image = np.pad(image, padding + [[0,0]])

        if mask is not None:
            mask = np.pad(mask, padding)
    
        H, W, C = image.shape

    image = image[
        (H-crop_size)//2:(H+crop_size)//2,
        (W-crop_size)//2:(W+crop_size)//2,
        :
    ]

    if mask is not None:
        mask = mask[
            (H-crop_size)//2:(H+crop_size)//2,
            (W-crop_size)//2:(W+crop_size)//2,
        ]

        return image, mask

    else:
        return image
