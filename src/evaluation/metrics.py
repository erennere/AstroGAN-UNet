"""Evaluation pipeline for model/image metrics and source catalogs."""

import os, glob, logging
import random
import sys
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import numpy as np
import pandas as pd

# Keep evaluation CPU-only when launched as its own entrypoint, but avoid
# hiding GPUs for training/config imports that only need symbols from here.
if __name__ == '__main__' and 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Ellipse 
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree  # type: ignore[attr-defined]
from scipy.ndimage import gaussian_filter

from astropy.visualization import MinMaxInterval, ZScaleInterval
from astropy.stats import SigmaClip, sigma_clipped_stats

import sep  # type: ignore[import-not-found]
from photutils.segmentation import detect_sources, deblend_sources, SourceCatalog, detect_threshold
from photutils.background import Background2D, SExtractorBackground
from photutils.utils import circular_footprint

from src.data.create_dataset import crop_image_generator
from src.training.new_train import data_augment_pluggable
from src.training.math_helpers import _simulated_image_from_exposure, _simulated_image_from_poisson, adaptive_log_transform_and_normalize, inverse_adaptive_log_transform_and_denormalize, min_max_normalization, inverse_min_max_normalization, zscore_normalization, inverse_zscore_normalization
from src.training.utils import open_fits, save_fits, ensure_directory_exists, ensure_parent_dir_exists, build_checkpoint_custom_objects, load_checkpoint_model, read_checkpoint_info
from starter import _decode_models_dir, load_config, parse_config_overrides  #sym:parse_config_overrides
logging.basicConfig(level=logging.WARNING)

def generate_gaussian_weights(patch_size, sigma=64):
    """Generate a 2-D Gaussian weight map for a sliding-window patch.

    Parameters
    ----------
    patch_size : tuple of int
        ``(height, width, channels)``. Only the first two elements are used.
    sigma : float, optional
        Standard deviation of the Gaussian in pixels. Default is 64.

    Returns
    -------
    numpy.ndarray or None
        Array of shape ``(height, width, 1)`` with Gaussian weights centred at
        the patch centre, or ``None`` if *patch_size* is invalid.
    """
    if patch_size is None or len(patch_size) != 3:
        logging.warning('generate_gaussian_weights: patch_size is not a tuple of length 3 but %s', patch_size)
        return
    h, w = patch_size[:2]  # Extract height and width, ignore depth
    ax_h = np.linspace(-(h // 2), h // 2, h)
    ax_w = np.linspace(-(w // 2), w // 2, w)
    xx, yy = np.meshgrid(ax_w, ax_h)  # Ensure correct shape order
    weights = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return weights[..., np.newaxis]  # Keep channel dimension

def generate_distance_weights(patch_size):
    """Generate a 2-D distance-based weight map for a sliding-window patch.

    Weights decrease linearly from 1.0 at the centre to 0.0 at the corners and
    are clipped to be non-negative.

    Parameters
    ----------
    patch_size : tuple of int
        ``(height, width, channels)``. Only the first two elements are used.

    Returns
    -------
    numpy.ndarray or None
        Array of shape ``(height, width, 1)`` with distance-based weights, or
        ``None`` if *patch_size* is invalid.
    """
    if patch_size is None or len(patch_size) != 3:
        logging.warning('generate_distance_weights: patch_size is not a tuple of length 3 but %s', patch_size)
        return
    h, w = patch_size[:2]
    ax_h = np.linspace(-1, 1, h)
    ax_w = np.linspace(-1, 1, w)
    xx, yy = np.meshgrid(ax_w, ax_h)
    weights = 1.0 - np.sqrt(xx**2 + yy**2)
    return np.clip(weights, 0, None)[..., np.newaxis]

def pad_image(image, patch_size, stride):
    """Pad a 3-D image so it is fully covered by non-overlapping patches.

    Two types of padding are applied:

    * **Overlap padding** (``pad_eq``) — adds ``patch_size - stride`` pixels on
      each side so that every pixel is visited at least once by a sliding
      window, even the border pixels.
    * **Divisibility padding** — pads the remaining margin so that the padded
      dimensions are exact multiples of ``patch_size``.

    Parameters
    ----------
    image : numpy.ndarray
        Input image of shape ``(H, W, C)``.
    patch_size : tuple of int
        ``(patch_H, patch_W, C)``.
    stride : tuple of int
        ``(stride_H, stride_W, C)``.

    Returns
    -------
    padded_image : numpy.ndarray
        Zero-padded image of shape
        ``(H + 2*pad_eq_h + pad_h, W + 2*pad_eq_w + pad_w, C)``.
    pad_amounts : tuple of int
        ``(pad_h_top, pad_h_bottom, pad_w_left, pad_w_right, pad_eq_h, pad_eq_w)``
        needed by :func:`strip_pad` to reverse the operation.
    """
    if patch_size is None or len(patch_size) != 3:
        logging.warning('pad_image: patch_size must be a 3-element tuple, got %s', patch_size)
        return

    if image is None:
        logging.warning('pad_image: image is None')
        return

    image_size = image.shape
    pad_h = int(np.ceil(image_size[0]/patch_size[0])*patch_size[0] - image_size[0])
    pad_w = int(np.ceil(image_size[1]/patch_size[1])*patch_size[1] - image_size[1])
    pad_h_top = pad_h//2
    pad_h_bottom = pad_h - pad_h_top
    pad_w_left = pad_w//2
    pad_w_right = pad_w - pad_w_left

    pad_eq_h = patch_size[0] - stride[0]
    pad_eq_w = patch_size[1] - stride[1]

    padded_eq_divisible_image = np.pad(
        image,
        ((pad_h_top + pad_eq_h, pad_h_bottom + pad_eq_h), (pad_w_left + pad_eq_w, pad_w_right + pad_eq_w), (0, 0)),
        mode='constant',
        constant_values=0
    )
    return padded_eq_divisible_image, (pad_h_top, pad_h_bottom, pad_w_left, pad_w_right, pad_eq_h, pad_eq_w)

def strip_pad(image, pad_h_top, pad_h_bottom, pad_w_left, pad_w_right, pad_eq_h, pad_eq_w):
    """Remove the padding added by :func:`pad_image`.

    Parameters
    ----------
    image : numpy.ndarray
        Padded image of shape ``(H_padded, W_padded, C)``.
    pad_h_top : int
        Pixels removed from the top (divisibility padding, top half).
    pad_h_bottom : int
        Pixels removed from the bottom (divisibility padding, bottom half).
    pad_w_left : int
        Pixels removed from the left (divisibility padding, left half).
    pad_w_right : int
        Pixels removed from the right (divisibility padding, right half).
    pad_eq_h : int
        Overlap-equalisation padding removed symmetrically top and bottom.
    pad_eq_w : int
        Overlap-equalisation padding removed symmetrically left and right.

    Returns
    -------
    numpy.ndarray or None
        Unpadded image with its original spatial dimensions, or ``None`` if the
        input is ``None`` or the padding exceeds the image size.
    """
    if image is None:
        logging.warning('strip_pad: image is None')
        return
    if  (pad_eq_h * 2 + pad_h_top + pad_h_bottom) >= image.shape[0]:
        logging.warning('strip_pad: image height %s smaller than total vertical padding', image.shape[0])
        return
    if  (pad_eq_w * 2 + pad_w_left + pad_w_right) >= image.shape[1]:
        logging.warning('strip_pad: image width %s smaller than total horizontal padding', image.shape[1])
        return

    if pad_eq_h > 0:
        image = image[pad_eq_h:-pad_eq_h, :, :]
    if pad_eq_w > 0:
        image = image[:, pad_eq_w:-pad_eq_w, :]
    if pad_h_top > 0:
        image = image[pad_h_top:, :, :]
    if pad_h_bottom > 0:
        image = image[:-pad_h_bottom, :, :]
    if pad_w_left > 0:
        image = image[:, pad_w_left:, :]
    if pad_w_right > 0:
        image = image[:, :-pad_w_right, :]
    return image

def sliding_window_generator(padded_image, patch_size=(256, 256, 1), stride=(128, 128, 1)):
    """Yield overlapping patches from a padded image in raster-scan order.

    Parameters
    ----------
    padded_image : numpy.ndarray
        Image of shape ``(H, W, C)`` — should already be padded by
        :func:`pad_image`.
    patch_size : tuple of int, optional
        ``(patch_H, patch_W, C)``. Default is ``(256, 256, 1)``.
    stride : tuple of int, optional
        ``(stride_H, stride_W, C)``. Default is ``(128, 128, 1)``.

    Yields
    ------
    patch : numpy.ndarray
        Array of shape ``(patch_H, patch_W, C)``.
    position : tuple of int
        ``(row, col)`` top-left corner of the patch in the padded image.
    """
    H, W, C = padded_image.shape
    
    for i in range(0, H - patch_size[0] + 1, stride[0]):
        for j in range(0, W - patch_size[1] + 1, stride[1]):
            patch = padded_image[i:i + patch_size[0], j:j + patch_size[1], :]
            yield patch, (i, j)  # Yield instead of appending

def create_prediction_dataset(image, patch_size=(256, 256, 1), stride=(128, 128, 1), batch_size=128):
    """Wrap :func:`sliding_window_generator` in a batched ``tf.data.Dataset``.

    Parameters
    ----------
    image : numpy.ndarray
        Image of shape ``(H, W, C)``.
    patch_size : tuple of int, optional
        ``(patch_H, patch_W, C)``. Default is ``(256, 256, 1)``.
    stride : tuple of int, optional
        ``(stride_H, stride_W, C)``. Default is ``(128, 128, 1)``.
    batch_size : int, optional
        Number of patches per batch fed to the model. Default is 128.

    Returns
    -------
    tf.data.Dataset
        Batched dataset yielding ``(patches, positions)`` tuples where
        *patches* has shape ``(B, patch_H, patch_W, C)`` and *positions*
        has shape ``(B, 2)``.
    """
    patch_generator = sliding_window_generator(image, patch_size, stride)
    dataset = tf.data.Dataset.from_generator(
        lambda: patch_generator,
        output_signature=(
            tf.TensorSpec(shape=(patch_size[0], patch_size[1], patch_size[2]), dtype=tf.float32), 
            tf.TensorSpec(shape=(2,), dtype=tf.int64)
            )
    )
    return dataset.batch(batch_size)

def sliding_window_inference(image, model, patch_size=(256, 256, 1),
                              stride=(128, 128, 1), weighting='average', batch_size=16,
                              gaussian_sigma=64):
    """Reconstruct an image using a model applied patch-by-patch with overlap blending.

    Parameters
    ----------
    image : numpy.ndarray
        Input image of shape ``(H, W, C)``.
    model : tf.keras.Model
        Trained model that accepts batches of patches.
    patch_size : tuple of int, optional
        ``(patch_H, patch_W, C)``. Default is ``(256, 256, 1)``.
    stride : tuple of int, optional
        Sliding stride ``(stride_H, stride_W, C)``. Default is ``(128, 128, 1)``.
    weighting : {'average', 'gaussian', 'distance'}, optional
        How overlapping patch contributions are weighted. Default is
        ``'average'``.
    batch_size : int, optional
        Patches per model call. Default is 16.
    gaussian_sigma : float, optional
        Sigma for the Gaussian weighting kernel (only used when
        ``weighting='gaussian'``). Default is 64.

    Returns
    -------
    numpy.ndarray or None
        Reconstructed image with the same shape as *image*, or ``None`` on
        failure.
    """
    image_shape = image.shape
    if any(p > s for p, s in zip(patch_size, image_shape)):
        logging.warning('sliding_window_inference: patch_size %s exceeds image size %s', patch_size, image_shape)
        return
    if any(st > s for st, s in zip(stride, image_shape)):
        logging.warning('sliding_window_inference: stride %s exceeds image size %s', stride, image_shape)
        return

    try:
        _pad_result = pad_image(image, patch_size, stride)
        if _pad_result is None:
            logging.warning('sliding_window_inference: pad_image returned None for image shape %s', image.shape)
            return
        image_fully_padded, (pad_h_top, pad_h_bottom, pad_w_left, pad_w_right, pad_eq_h, pad_eq_w) = _pad_result
    except Exception as err:
        logging.warning(f"could not unpack 'pad_image': {err}")
        return
    
    output = np.zeros_like(image_fully_padded, dtype=np.float32)
    weight_matrix = np.zeros_like(image_fully_padded, dtype=np.float32)

    if weighting == 'gaussian':
        weights = generate_gaussian_weights(patch_size, gaussian_sigma)
        if weights is None:
            return 
    elif weighting == 'distance':
        weights = generate_distance_weights(patch_size)
        if weights is None:
            return 
    elif weighting == 'average':
        weights = np.ones(patch_size, dtype=np.float32)
        if weights is None:
            return 
    else:
        logging.warning("sliding_window_inference: unknown weighting '%s', must be 'average', 'gaussian', or 'distance'", weighting)
        return
    
    try:
        dataset = create_prediction_dataset(image_fully_padded, patch_size, stride, batch_size)
        for patches, positions in dataset:
            batch_predictions = model.predict(patches)  # Get predictions from the model
            for (i, j), prediction in zip(positions, batch_predictions):
                output[i:i + patch_size[0], j:j + patch_size[1], :] += prediction * weights
                weight_matrix[i:i + patch_size[0], j:j + patch_size[1], :] += weights

        output = strip_pad(output, pad_h_top, pad_h_bottom, pad_w_left, pad_w_right, pad_eq_h, pad_eq_w)
        weight_matrix = strip_pad(weight_matrix, pad_h_top, pad_h_bottom, pad_w_left, pad_w_right, pad_eq_h, pad_eq_w)
        if output is None or weight_matrix is None:
            logging.warning('sliding_window_inference: strip_pad returned None')
            return
    except Exception as err:
        logging.warning(f"an error occurred while stripping padding: {err}")
        return
    return output / (weight_matrix + 1e-12)

def scale_image(image, vmin=None, vmax=None):
    """Convert a floating-point image array to a PIL greyscale image.

    Applies ZScale normalisation (or user-supplied limits) and maps pixel
    values to the ``[0, 255]`` range.

    Parameters
    ----------
    image : numpy.ndarray
        2-D (or 3-D with a single channel) numerical array.
    vmin : float or None, optional
        Lower display limit. If ``None`` the ZScale lower limit is used.
    vmax : float or None, optional
        Upper display limit. If ``None`` the ZScale upper limit is used.

    Returns
    -------
    img : PIL.Image.Image
        8-bit greyscale image.
    vmin : float
        Lower display limit actually used.
    vmax : float
        Upper display limit actually used.
    """
    try:
        if not isinstance(image, np.ndarray):
            logging.warning('scale_image: expected a NumPy array, got %s', type(image))
            return
        if image.ndim < 2:
            logging.warning('scale_image: expected at least a 2D image, got shape %s', image.shape)
            return
        if not np.issubdtype(image.dtype, np.number):
            logging.warning('scale_image: image array must contain numerical values, got dtype %s', image.dtype)
            return

        if vmin is None or vmax is None:
            interval = ZScaleInterval()
            vmin, vmax = interval.get_limits(image)

        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        image_2d = image.squeeze() if image.ndim == 3 and image.shape[2] == 1 else image
        normalized_data = (norm(image_2d) * 255).astype(np.uint8)
        img = Image.fromarray(normalized_data)
    except Exception as err:
        logging.warning(f"No scaling of images was possible: {err}, returning the original image.")
        return image, vmin, vmax
    return img, vmin, vmax

def plot_source_comparison(original_image, noisy_image, reconstructed_image, 
                           x_org, y_org, x_rec, y_rec, 
                           matched_indices_org, unmatched_indices_org, 
                           matched_indices_rec, unmatched_indices_rec,
                           filepath):
    """Save a three-panel comparison plot of detected sources.

    Panels show the original, noisy, and reconstructed images with matched
    (red) and unmatched (blue/green) source positions overlaid as circular
    scatter markers.

    Parameters
    ----------
    original_image : numpy.ndarray
        2-D reference image array.
    noisy_image : numpy.ndarray
        2-D noisy input image array.
    reconstructed_image : numpy.ndarray
        2-D model-reconstructed image array.
    x_org, y_org : numpy.ndarray
        Source centroid x/y coordinates in the original image.
    x_rec, y_rec : numpy.ndarray
        Source centroid x/y coordinates in the reconstructed image.
    matched_indices_org : numpy.ndarray
        Indices into *x_org* / *y_org* for matched sources.
    unmatched_indices_org : numpy.ndarray
        Indices into *x_org* / *y_org* for unmatched (false-negative) sources.
    matched_indices_rec : numpy.ndarray
        Indices into *x_rec* / *y_rec* for matched sources.
    unmatched_indices_rec : numpy.ndarray
        Indices into *x_rec* / *y_rec* for unmatched (false-positive) sources.
    filepath : str
        Destination path for the saved PNG file.
    """
    try:
        for image in [original_image, noisy_image, reconstructed_image]:
            if not isinstance(image, np.ndarray):
                logging.warning('plot_source_comparison: expected a NumPy array, got %s', type(image))
                return
            if image.ndim < 2:
                logging.warning('plot_source_comparison: expected at least a 2D image, got shape %s', image.shape)
                return
            if not np.issubdtype(image.dtype, np.number):
                logging.warning('plot_source_comparison: image array must contain numerical values, got dtype %s', image.dtype)
                return

        try:
            original_image_scaled, vmin, vmax = scale_image(original_image)
            noisy_image_scaled, _, _ = scale_image(noisy_image, vmin=vmin, vmax=vmax)
            reconstructed_image_scaled, _, _ = scale_image(reconstructed_image, vmin=vmin, vmax=vmax)
        except Exception as err:
            logging.warning(f'an error occurred while scaling images: {err}')
            return

        plt.figure(figsize=(18, 6))

        # Original Image
        plt.subplot(1, 3, 1)
        plt.imshow(original_image_scaled, cmap='gray')
        plt.scatter(x_org[unmatched_indices_org], y_org[unmatched_indices_org], 
                    edgecolors='blue', facecolors='none', label='Unmatched Sources (Original)')
        plt.scatter(x_org[matched_indices_org], y_org[matched_indices_org], 
                    edgecolors='red', facecolors='none', label='Matched Sources')
        plt.title('Detected Sources in Original Image')
        plt.legend()

        # Noisy Image (Just Plot)
        plt.subplot(1, 3, 2)
        plt.imshow(noisy_image_scaled, cmap='gray')
        plt.title('Noisy Image')

        # Reconstructed Image
        plt.subplot(1, 3, 3)
        plt.imshow(reconstructed_image_scaled, cmap='gray')
        plt.scatter(x_rec[unmatched_indices_rec], y_rec[unmatched_indices_rec], 
                    edgecolors='green', facecolors='none', label='Unmatched Sources (Reconstructed)')
        plt.scatter(x_rec[matched_indices_rec], y_rec[matched_indices_rec], 
                    edgecolors='red', facecolors='none', label='Matched Sources')
        plt.title('Detected Sources in Reconstructed Image')
        plt.legend()

        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as err:
        logging.warning(f"an error occured while creating combined images: {err}")

def plot_source_comparison_sep(original_image, noisy_image, reconstructed_image, 
                           x_org, y_org, x_rec, y_rec, 
                           matched_indices_org, unmatched_indices_org, 
                           matched_indices_rec, unmatched_indices_rec,
                           org_a, org_b, rec_a, rec_b, org_theta, rec_theta,
                           filepath):
    """Save a three-panel source comparison plot with SEP ellipse apertures.

    Same layout as :func:`plot_source_comparison` but draws SEP Kron ellipses
    around each source instead of circular scatter markers.

    Parameters
    ----------
    original_image : numpy.ndarray
        2-D reference image array.
    noisy_image : numpy.ndarray
        2-D noisy input image array.
    reconstructed_image : numpy.ndarray
        2-D model-reconstructed image array.
    x_org, y_org : numpy.ndarray
        Source centroid x/y coordinates in the original image.
    x_rec, y_rec : numpy.ndarray
        Source centroid x/y coordinates in the reconstructed image.
    matched_indices_org : numpy.ndarray
        Indices into the original-image source arrays for matched sources.
    unmatched_indices_org : numpy.ndarray
        Indices into the original-image source arrays for unmatched sources.
    matched_indices_rec : numpy.ndarray
        Indices into the reconstructed-image source arrays for matched sources.
    unmatched_indices_rec : numpy.ndarray
        Indices into the reconstructed-image source arrays for unmatched sources.
    org_a, org_b : numpy.ndarray
        Semi-major and semi-minor axes of original-image sources (pixels).
    rec_a, rec_b : numpy.ndarray
        Semi-major and semi-minor axes of reconstructed-image sources (pixels).
    org_theta, rec_theta : numpy.ndarray
        Position angles (radians) of original and reconstructed sources.
    filepath : str
        Destination path for the saved PNG file.
    """
    try:
        for image in [original_image, noisy_image, reconstructed_image]:
            if not isinstance(image, np.ndarray):
                logging.warning('plot_source_comparison_sep: expected a NumPy array, got %s', type(image))
                return
            if image.ndim < 2:
                logging.warning('plot_source_comparison_sep: expected at least a 2D image, got shape %s', image.shape)
                return
            if not np.issubdtype(image.dtype, np.number):
                logging.warning('plot_source_comparison_sep: image array must contain numerical values, got dtype %s', image.dtype)
                return
    
        try:
            original_image_scaled, vmin, vmax = scale_image(original_image)
            noisy_image_scaled, _, _ = scale_image(noisy_image, vmin=vmin, vmax=vmax)
            reconstructed_image_scaled, _, _ = scale_image(reconstructed_image, vmin=vmin, vmax=vmax)
        except Exception as err:
            logging.warning(f"An error occurred while scaling images: {err}")
            return

        plt.figure(figsize=(18, 6))

        # Original Image
        ax1 = plt.subplot(1, 3, 1)
        ax1.imshow(original_image_scaled, cmap='gray')

        for x, y, a, b, theta in zip(x_org[unmatched_indices_org], y_org[unmatched_indices_org], 
                                      org_a[unmatched_indices_org], org_b[unmatched_indices_org], 
                                      org_theta[unmatched_indices_org]):
            e = Ellipse(xy=(x, y), width=6*a, height=6*b, angle=np.degrees(theta), 
                        edgecolor='blue', facecolor='none', linewidth=1.5)
            ax1.add_patch(e)

        for x, y, a, b, theta in zip(x_org[matched_indices_org], y_org[matched_indices_org], 
                                      org_a[matched_indices_org], org_b[matched_indices_org], 
                                      org_theta[matched_indices_org]):
            e = Ellipse(xy=(x, y), width=6*a, height=6*b, angle=np.degrees(theta), 
                        edgecolor='red', facecolor='none', linewidth=1.5)
            ax1.add_patch(e)

        ax1.set_title('Detected Sources in the Original Image')

        # Noisy Image (Only Image)
        ax2 = plt.subplot(1, 3, 2)
        ax2.imshow(noisy_image_scaled, cmap='gray')
        ax2.set_title('Noisy Image')

        # Reconstructed Image
        ax3 = plt.subplot(1, 3, 3)
        ax3.imshow(reconstructed_image_scaled, cmap='gray')

        for x, y, a, b, theta in zip(x_rec[unmatched_indices_rec], y_rec[unmatched_indices_rec], 
                                      rec_a[unmatched_indices_rec], rec_b[unmatched_indices_rec], 
                                      rec_theta[unmatched_indices_rec]):
            e = Ellipse(xy=(x, y), width=6*a, height=6*b, angle=np.degrees(theta), 
                        edgecolor='green', facecolor='none', linewidth=1.5)
            ax3.add_patch(e)

        for x, y, a, b, theta in zip(x_rec[matched_indices_rec], y_rec[matched_indices_rec], 
                                      rec_a[matched_indices_rec], rec_b[matched_indices_rec], 
                                      rec_theta[matched_indices_rec]):
            e = Ellipse(xy=(x, y), width=6*a, height=6*b, angle=np.degrees(theta), 
                        edgecolor='red', facecolor='none', linewidth=1.5)
            ax3.add_patch(e)

        ax3.set_title('Detected Sources in the Reconstructed Image')

        # Custom Legend
        legend_elements = [
            Line2D([0], [0], color='red', lw=2, label='Matched Sources'),
            Line2D([0], [0], color='blue', lw=2, label='Unmatched Sources (Original)'),
            Line2D([0], [0], color='green', lw=2, label='Unmatched Sources (Reconstructed)')
        ]
        
        plt.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.2),
                   ncol=3, frameon=False)

        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

    except Exception as err:
        logging.warning(f"An error occurred while creating combined images: {err}")
        
def compute_ssim(x, y, alpha=1, beta=1, gamma=1, k1=0.01, k2=0.03, win_size=11, win_sigma=1.5):
    """Compute the local (sliding-window) SSIM following Wang et al. 2004.

    Parameters
    ----------
    x : numpy.ndarray
        Reference image array (2-D or 3-D with a single trailing channel).
    y : numpy.ndarray
        Distorted image array; must have the same shape as *x*.
    alpha : float, optional
        Exponent for the luminance component. Default is 1.
    beta : float, optional
        Exponent for the contrast component. Default is 1.
    gamma : float, optional
        Exponent for the structure component. Default is 1.
    k1 : float, optional
        Stability constant for luminance (typical 0.01). Default is 0.01.
    k2 : float, optional
        Stability constant for contrast (typical 0.03). Default is 0.03.
    win_size : int, optional
        Odd integer: Gaussian kernel width in pixels. Standard value is 11.
        Rule of thumb: 1–2 % of the shorter image dimension, rounded to the
        nearest odd number. Default is 11.
    win_sigma : float, optional
        Standard deviation of the Gaussian kernel. Standard value is 1.5.
        The truncation radius equals ``win_size // 2``. Default is 1.5.

    Returns
    -------
    mean_ssim : float
        Mean SSIM value over all valid pixels (NaN values are ignored).
    ssim_map : numpy.ndarray
        Pixel-wise SSIM map with the same shape as *x*.
    """
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or x.shape != y.shape:
        return None, None

    # Work on 2D float64 arrays; squeeze single channel dim if present
    squeezed = x.ndim == 3 and x.shape[2] == 1
    if squeezed:
        x2d = x[:, :, 0].astype(np.float64)
        y2d = y[:, :, 0].astype(np.float64)
    else:
        x2d = x.astype(np.float64)
        y2d = y.astype(np.float64)

    L = np.max(x2d) - np.min(x2d)
    if L == 0:
        return np.nan, np.full_like(x, np.nan)

    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2
    c3 = c2 / 2

    # Gaussian-weighted local statistics (Wang et al. 2004)
    # truncate is chosen so the kernel radius equals win_size // 2
    truncate = (win_size // 2) / win_sigma
    filt = lambda a: gaussian_filter(a, sigma=win_sigma, truncate=truncate)

    mu_x = filt(x2d)
    mu_y = filt(y2d)
    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2

    # Clamp to 0 to avoid sqrt of tiny negative values from floating-point
    sigma_x_sq = np.maximum(filt(x2d ** 2) - mu_x_sq, 0)
    sigma_y_sq = np.maximum(filt(y2d ** 2) - mu_y_sq, 0)
    sigma_xy   = filt(x2d * y2d) - mu_x * mu_y

    sigma_x = np.sqrt(sigma_x_sq)
    sigma_y = np.sqrt(sigma_y_sq)

    # Luminance, contrast, structure components
    l = (2 * mu_x * mu_y + c1) / (mu_x_sq + mu_y_sq + c1)
    c = (2 * sigma_x * sigma_y + c2) / (sigma_x_sq + sigma_y_sq + c2)
    s = (sigma_xy + c3) / (sigma_x * sigma_y + c3)

    ssim_map = (l ** alpha) * (c ** beta) * (s ** gamma)
    if squeezed:
        ssim_map = ssim_map[:, :, np.newaxis]
    return float(np.nanmean(ssim_map)), ssim_map

def calculate_psnr(original, noisy):
    """Calculate the Peak Signal-to-Noise Ratio between two images.

    The dynamic range ``L`` is computed as ``max(original) - min(original)`` so
    that background-subtracted images with negative pixel values are handled
    correctly.

    Parameters
    ----------
    original : numpy.ndarray
        Reference image array.
    noisy : numpy.ndarray
        Distorted image array; must have the same shape as *original*.

    Returns
    -------
    psnr : float
        PSNR value in dB, or ``nan`` when the dynamic range is zero, or
        ``inf`` when the images are identical.
    mse : float
        Mean squared error between *original* and *noisy*.
    """
    # Ensure the images are of the same shape
    if not isinstance(original, np.ndarray) or not isinstance(noisy, np.ndarray) or original.shape != noisy.shape:
        logging.warning('calculate_psnr: input images must be NumPy arrays with identical shapes')
        return None, None
    # Dynamic range: max - min handles background-subtracted (negative-valued) images
    L = np.max(original) - np.min(original)
    if L == 0:
        logging.warning('calculate_psnr: original image has zero dynamic range, PSNR is undefined')
        return np.nan, np.nan
    # Calculate MSE
    mse = np.mean((original - noisy) ** 2)
    if mse == 0:
        return float('inf'), mse  # Identical images
    psnr = 10 * np.log10((L ** 2) / mse)
    return psnr, mse

def aggregate_df(df, flux_rec, flux_org, flux_error_rec, flux_error_org):
    """Aggregate per-image metrics into a single-row summary dictionary.

    PSNR is aggregated by averaging the per-image MSEs first and then
    converting to dB (averaging dB values directly is statistically incorrect).

    Parameters
    ----------
    df : pandas.DataFrame
        Per-image metrics DataFrame with columns including ``'TP'``, ``'FP'``,
        ``'FN'``, ``'MSE_rec'``, ``'MSE_noisy'``, ``'PSNR_L'``,
        ``'PSNR_rec'``, ``'PSNR_noisy'``, ``'SSIM_rec'``, ``'SSIM_noisy'``,
        and ``'IoU'``.
    flux_rec : numpy.ndarray
        Matched reconstructed-source flux values.
    flux_org : numpy.ndarray
        Matched original-source flux values.
    flux_error_rec : numpy.ndarray
        Flux uncertainties for the reconstructed sources.
    flux_error_org : numpy.ndarray
        Flux uncertainties for the original sources.

    Returns
    -------
    dict or None
        Dictionary of aggregated statistics with keys ``'TP'``, ``'FP'``,
        ``'FN'``, ``'Precision'``, ``'Recall'``, ``'F-measure'``, ``'RFE'``,
        ``'SNR_org'``, ``'SNR_rec'``, ``'PSNR_rec'``, ``'PSNR_noisy'``,
        ``'SSIM_rec'``, ``'SSIM_noisy'``, and ``'IoU'``, or ``None`` if *df*
        is not a DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        logging.warning('aggregate_df: expected a Pandas DataFrame, got %s', type(df))
        return

    tp = df.get('TP', pd.Series(0)).sum()
    fp = df.get('FP', pd.Series(0)).sum()
    fn = df.get('FN', pd.Series(0)).sum()

    # Aggregate PSNR correctly: average MSE first, then convert (averaging dB directly is wrong)
    mean_mse_rec = df.get('MSE_rec', pd.Series(np.nan)).mean()
    mean_mse_noisy = df.get('MSE_noisy', pd.Series(np.nan)).mean()
    L_rec = df.get('PSNR_L', pd.Series(np.nan)).mean()  # fallback: use stored L if available
    
    # Use per-image PSNR mean only as fallback when MSE is unavailable
    psnr_rec = df.get('PSNR_rec', pd.Series(np.nan)).mean()
    psnr_noisy = df.get('PSNR_noisy', pd.Series(np.nan)).mean()
    if not np.isnan(mean_mse_rec) and mean_mse_rec > 0:
        # Recompute from pooled MSE; use median L across images as representative dynamic range
        L = df.get('PSNR_L', pd.Series(np.nan)).median()
        if not np.isnan(L) and L > 0:
            psnr_rec = 10 * np.log10((L ** 2) / mean_mse_rec)
    if not np.isnan(mean_mse_noisy) and mean_mse_noisy > 0:
        L = df.get('PSNR_L', pd.Series(np.nan)).median()
        if not np.isnan(L) and L > 0:
            psnr_noisy = 10 * np.log10((L ** 2) / mean_mse_noisy)

    ssim_rec = df.get('SSIM_rec', pd.Series(np.nan)).mean()
    ssim_noisy = df.get('SSIM_noisy', pd.Series(np.nan)).mean()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f_measure = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    if isinstance(flux_org, np.ndarray) and isinstance(flux_rec, np.ndarray) and flux_org.shape == flux_rec.shape:
        nonzero = flux_org != 0.0
        rfe = np.mean((flux_org[nonzero] - flux_rec[nonzero]) / flux_org[nonzero]) if nonzero.any() else np.nan
    else:
        rfe = np.nan
    if isinstance(flux_rec, np.ndarray) and isinstance(flux_error_rec, np.ndarray) and flux_rec.shape == flux_error_rec.shape:
        snr_rec = np.mean(flux_rec / flux_error_rec)
    else:
        snr_rec = 0
    if isinstance(flux_org, np.ndarray) and isinstance(flux_error_org, np.ndarray) and flux_org.shape == flux_error_org.shape:
        snr_org = np.mean(flux_org / flux_error_org)
    else:
        snr_org = 0

    stats = {
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'Precision': precision,
        'Recall': recall,
        'F-measure': f_measure,
        'RFE': rfe,
        'SNR_org': snr_org,
        'SNR_rec': snr_rec,
        'PSNR_rec': psnr_rec,
        'PSNR_noisy': psnr_noisy,
        'SSIM_rec': ssim_rec,
        'SSIM_noisy': ssim_noisy,
        'IoU' : df['IoU'].mean()
    }
    return stats

def detect_sources_in_image(image, kwargs):
    """Detect, deblend, and characterise sources using photutils.

    Parameters
    ----------
    image : numpy.ndarray
        2-D flux image.
    kwargs : dict
        Detection and deblending parameters. Required keys:

        * ``'sigma'`` — sigma-clipping sigma for background estimation.
        * ``'maxiters'`` — maximum sigma-clipping iterations.
        * ``'nsigma'`` — detection threshold in units of background RMS.
        * ``'npixels'`` — minimum connected pixels for source detection.
        * ``'nlevels'`` — number of deblending threshold levels.
        * ``'contrast'`` — minimum contrast ratio for deblending.
        * ``'footprint_radius'`` — radius of the circular footprint for the
          source mask.
        * ``'deblend'`` — ``bool``; whether to apply deblending.
        * ``'deblend_timeout'`` — timeout in seconds for the deblending step.

    Returns
    -------
    x_centroids : numpy.ndarray
        Source x-centroid positions.
    y_centroids : numpy.ndarray
        Source y-centroid positions.
    fluxes : numpy.ndarray
        Segment flux values for each source.
    flux_errors : numpy.ndarray
        Flux uncertainties (background RMS times square root of segment area).
    mask : numpy.ndarray of bool
        Boolean source mask (``True`` where a source is present).

    Returns ``None`` if required parameters are missing, the image is invalid,
    or no sources are detected.
    """
    required_params = ['sigma', 'maxiters', 'nsigma',
                        'npixels', 'nlevels', 'contrast',
                        'footprint_radius', 'deblend', 'deblend_timeout']
    missing_params = [param for param in required_params if param not in kwargs]
    if missing_params:
        logging.warning('detect_sources_in_image: missing required parameters: %s', ', '.join(missing_params))
        return 
    if not isinstance(image, np.ndarray) or image.ndim != 2:
        logging.warning('detect_sources_in_image: image must be a 2D NumPy array, got shape %s', getattr(image, 'shape', type(image)))
        return 

    bkg = None
    try:
        sigma_clip = SigmaClip(sigma=kwargs['sigma'], maxiters=kwargs['maxiters'])
        try:
            bkg = Background2D(image, box_size=kwargs['bkg_box_size'], sigma_clip=sigma_clip,  # type: ignore[arg-type]
                           bkg_estimator=SExtractorBackground())
            threshold = bkg.background + kwargs['nsigma'] * bkg.background_rms
        except Exception as err:
            logging.warning('detect_sources_in_image: error during background estimation: %s', err)
            threshold = detect_threshold(image, n_sigma=kwargs['nsigma'], sigma_clip=sigma_clip)  # type: ignore[arg-type]

        segment_img = detect_sources(image, threshold, n_pixels=kwargs['npixels'])
        if segment_img is None:
            logging.warning('detect_sources_in_image: detect_sources returned None, no sources detected')
            return None  # No sources detected

        if kwargs['deblend']:
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(deblend_sources, image, segment_img,
                                            n_pixels=kwargs['npixels'],
                                            n_levels=kwargs['nlevels'],
                                            contrast=kwargs['contrast'])
                    try:
                        segm_deblended = future.result(timeout=kwargs['deblend_timeout'])
                    except FuturesTimeoutError:
                        logging.warning('detect_sources_in_image: deblending timed out, using undeblended segmentation')
                        segm_deblended = segment_img
            except Exception as err:
                logging.warning('detect_sources_in_image: error during deblending: %s', err)
                segm_deblended = segment_img
        else:
            segm_deblended = segment_img

        if segm_deblended is None:
            logging.warning('No sources detected; returning')
            return None
        if not hasattr(segm_deblended, 'make_source_mask'):
            logging.warning('Segmentation object does not expose make_source_mask; returning')
            return None
        
        footprint = circular_footprint(radius=kwargs['footprint_radius'])
        mask = segm_deblended.make_source_mask(footprint=footprint) # type: ignore[union-attr]
        mask = np.asarray(mask, dtype=bool)

        background_map = None
        error_map = None
        if bkg is None:
            _mean_bkg, median_bkg, std_bkg = sigma_clipped_stats(
                image,
                sigma=kwargs['sigma'],
                mask=mask,
            )
            background_map = np.full(image.shape, median_bkg, dtype=float)
            error_map = np.full(image.shape, std_bkg, dtype=float)
            background_subtracted_image = image - background_map
        else:
            background_map = np.asarray(bkg.background, dtype=float)
            error_map = np.asarray(bkg.background_rms, dtype=float)
            background_subtracted_image = image - bkg.background

        catalog = SourceCatalog(
            background_subtracted_image,
            segm_deblended,
            error=error_map,
            background=background_map,
        )
        if len(catalog) == 0:
            logging.warning('detect_sources_in_image: no sources found in catalog after deblending')
            return None

        x_centroids = np.array([source.xcentroid for source in catalog])
        y_centroids = np.array([source.ycentroid for source in catalog])
        fluxes = np.array([source.segment_flux for source in catalog])
        flux_errors = np.array([source.segment_flux_err for source in catalog])
        return x_centroids, y_centroids, fluxes, flux_errors, mask
    except Exception as err:
        logging.warning('detect_sources_in_image: error during source detection: %s', err)
        return
    
def extract_sources(image, image_flag, kwargs):
    """Extract sources from an astronomical image using SEP and classify them.

    Background is estimated and subtracted with SEP, then sources are extracted,
    Kron fluxes computed, and each source is labelled as a galaxy based on its
    axis-ratio elongation.

    Parameters
    ----------
    image : numpy.ndarray
        2-D floating-point image in native byte order.
    image_flag : str
        One of ``'original'``, ``'reconstructed'``, or ``'noisy'``. Controls
        which threshold/minarea parameter is read from *kwargs* (original images
        use ``'org_thresh'`` / ``'org_minarea'``).
    kwargs : dict
        SEP extraction and photometry parameters. Recognised keys include:
        ``'thresh'``, ``'org_thresh'``, ``'radius_factor'``, ``'PHOT_FLUXFRAC'``,
        ``'r_min'``, ``'elongation_fraction'``, ``'PHOT_AUTOPARAMS'``,
        ``'maskthresh'``, ``'minarea'``, ``'org_minarea'``, ``'filter_type'``,
        ``'deblend_nthresh'``, ``'deblend_cont'``, ``'clean'``,
        ``'clean_param'``.

    Returns
    -------
    merged_df : pandas.DataFrame
        One row per detected source with SEP output columns plus
        ``'kron_radius'``, ``'flux'``, ``'flux_err'``, ``'flux_radius'``,
        ``'sep_flag'``, and ``'is_galaxy'``.
    mask : numpy.ndarray of bool
        Boolean ellipse mask marking source pixels.
    """
    # Get parameters from kwargs or set defaults

    thresh = kwargs['thresh'] if image_flag != 'original' else kwargs['org_thresh']
    radius_factor = kwargs['radius_factor']  # Factor for Kron radius
    PHOT_FLUXFRAC  = kwargs['PHOT_FLUXFRAC']  # Fraction for flux radius
    r_min = kwargs['r_min']  # Minimum radius for circular apertures
    elongation_fraction = kwargs['elongation_fraction']
    PHOT_AUTOPARAMS = kwargs['PHOT_AUTOPARAMS']

    maskthresh = kwargs['maskthresh']  # Threshold for pixel masking
    minarea = kwargs['minarea'] if image_flag != 'original' else kwargs['org_minarea']
    filter_type = kwargs['filter_type']  # Filter treatment type
    deblend_nthresh = kwargs['deblend_nthresh']  # Number of thresholds for deblending
    deblend_cont = kwargs['deblend_cont']  # Minimum contrast ratio for deblending
    clean = kwargs['clean']  # Perform cleaning
    clean_param = kwargs['clean_param']  # Cleaning parameter

    # Subtract background using SEP
    image = image.astype(image.dtype.newbyteorder('='))  # Converts to the native byte order
    bkg = sep.Background(image)
    data_sub = np.ascontiguousarray(image - bkg).copy()

    objects = sep.extract(data_sub, thresh, err=bkg.rms(), maskthresh=maskthresh, minarea=minarea, 
                          filter_type=filter_type, deblend_nthresh=deblend_nthresh, deblend_cont=deblend_cont, 
                          clean=clean, clean_param=clean_param)
    
    objects = pd.DataFrame(objects)
    objects['index'] = range(len(objects))
    x = objects['x'].to_numpy(dtype=float, copy=True)
    y = objects['y'].to_numpy(dtype=float, copy=True)
    a = objects['a'].to_numpy(dtype=float, copy=True)
    b = objects['b'].to_numpy(dtype=float, copy=True)
    theta = objects['theta'].to_numpy(dtype=float, copy=True)

    rms_map = bkg.rms()

    # Calculate Kron radius
    kronrad, krflag = sep.kron_radius(data_sub, x, y, a, b, theta, radius_factor)
    
    # Calculate flux using elliptical aperture
    flux, fluxerr, phot_flag = sep.sum_ellipse(data_sub, x, y, a, b, theta,
                                               PHOT_AUTOPARAMS*kronrad, subpix=1, err=rms_map)
    phot_flag |= krflag

    # Use circular aperture if Kron radius is small
    use_circle = kronrad * np.sqrt(a * b) < r_min
    cflux, cfluxerr, cflag = sep.sum_circle(data_sub, x[use_circle], y[use_circle], r_min, subpix=1, err=rms_map)
    flux[use_circle] = cflux
    fluxerr[use_circle] = cfluxerr
    phot_flag[use_circle] = cflag

    # Compute flux radius
    r, rflag = sep.flux_radius(data_sub, x, y, radius_factor * a, PHOT_FLUXFRAC, normflux=flux, subpix=5)
    phot_flag |= rflag
    
    # Create mask
    mask = np.zeros(data_sub.shape, dtype=bool)
    sep.mask_ellipse(mask, x, y, a, b, theta, r=3.)

    # Prepare results
    results = pd.DataFrame({
        'index': objects['index'],
        'kron_radius': kronrad,
        'flux': flux,
        'flux_err': fluxerr,
        'flux_radius': r,
        'sep_flag': phot_flag
    })

    merged_df = pd.merge(objects, results, on="index").drop(columns=['index'])
    merged_df['is_galaxy'] = (merged_df['a'] / merged_df['b'] >= elongation_fraction).astype(int)
    return merged_df, mask

def wrap_extract_sources(image, flag, kwargs):
    """Thin wrapper around :func:`extract_sources` that returns flat arrays.

    Unpacks the DataFrame returned by :func:`extract_sources` into individual
    NumPy arrays for direct use in :func:`compare_images`.

    Parameters
    ----------
    image : numpy.ndarray
        2-D flux image.
    flag : str
        Image type flag passed to :func:`extract_sources` (e.g.
        ``'original'``, ``'reconstructed'``, ``'noisy'``).
    kwargs : dict
        Parameter dictionary forwarded to :func:`extract_sources`.

    Returns
    -------
    x : numpy.ndarray
        Source x-centroid positions.
    y : numpy.ndarray
        Source y-centroid positions.
    flux : numpy.ndarray
        Source flux values.
    flux_err : numpy.ndarray
        Source flux uncertainties.
    mask : numpy.ndarray of bool
        Boolean source mask.
    a, b : numpy.ndarray
        Semi-major and semi-minor axis lengths (pixels).
    theta : numpy.ndarray
        Position angles (radians).
    df : pandas.DataFrame
        Full SEP source catalogue for this image.
    """
    df, mask = extract_sources(image, flag, kwargs)
    return df['x'].to_numpy(), df['y'].to_numpy(), df['flux'].to_numpy(), df['flux_err'].to_numpy(), mask, df['a'].to_numpy(), df['b'].to_numpy(), df['theta'].to_numpy(), df

def calculate_iou(org_mask, rec_mask):
    """Calculate Intersection over Union (IoU) for two binary source masks.

    Parameters
    ----------
    org_mask : numpy.ndarray
        Binary mask from the original image (``True``/1 for source pixels,
        ``False``/0 for background).
    rec_mask : numpy.ndarray
        Binary mask from the reconstructed image; must have the same shape as
        *org_mask*.

    Returns
    -------
    iou : float
        Intersection over Union score in ``[0, 1]``, or ``nan`` when the union
        is zero or the masks have different shapes.
    union : int or float
        Number of pixels in the union of the two masks, or ``nan`` on shape
        mismatch.
    """
    # Ensure that the masks are binary (values should be either 0 or 1)
    org_mask = np.array(org_mask, dtype=bool)
    rec_mask = np.array(rec_mask, dtype=bool)
    if org_mask.shape != rec_mask.shape:
        return np.nan, np.nan

    intersection = np.sum(org_mask & rec_mask)
    union = np.sum(org_mask | rec_mask)
    iou = intersection / union if union != 0 else np.nan
    return iou, union

def compare_images(image_org, noisy_image, image_reconstructed, image_id, exp_time,
                    new_exp_time, output_dir, kwargs, if_selected):
    """Compute per-image quality metrics and source-matching statistics.

    Detects sources in all three images (original, noisy, reconstructed) using
    the callable in ``kwargs['func']``, matches original to reconstructed
    sources with a KD-tree, and returns detection/flux/image-quality metrics.

    Parameters
    ----------
    image_org : numpy.ndarray
        2-D reference (original) image array.
    noisy_image : numpy.ndarray
        2-D noisy (degraded) image array.
    image_reconstructed : numpy.ndarray
        2-D model-reconstructed image array.
    image_id : str
        Unique identifier for this image (used in output filepaths).
    exp_time : float
        Original exposure time (seconds).
    new_exp_time : float
        Simulated (reduced) exposure time (seconds).
    output_dir : str
        Directory where the source-comparison PNG is saved.
    kwargs : dict
        Configuration dictionary containing source-detection parameters,
        SSIM parameters, and the callable under key ``'func'``
        (either :func:`wrap_extract_sources` or :func:`detect_sources_in_image`).
    if_selected : bool
        Whether to save the comparison plot for this image.

    Returns
    -------
    stats : dict
        Per-image statistics including detection counts, PSNR, SSIM, RFE,
        SNR, IoU, and exposure times.
    flux_arrays : tuple
        ``(flux_rec, flux_org)`` — matched reconstructed and original flux
        values as flat lists.
    flux_error_arrays : tuple
        ``(flux_error_rec, flux_error_org)`` — matched flux uncertainties as
        flat lists.
    dfs : tuple
        ``(org_df, noisy_df, rec_df)`` — per-image source catalogues as
        DataFrames (empty when the photutils path is used).

    Returns ``None`` if the input images are not compatible NumPy arrays or if
    source detection fails.
    """
    if not isinstance(image_org, np.ndarray) or not isinstance(noisy_image, np.ndarray) or not isinstance(image_reconstructed, np.ndarray):
        logging.warning('compare_images: all input images must be NumPy arrays: got types %s, %s, %s', type(image_org), type(noisy_image), type(image_reconstructed))
        return
    if image_org.shape != noisy_image.shape or noisy_image.shape != image_reconstructed.shape or image_org.shape != image_reconstructed.shape:
        logging.warning('compare_images: all input images must have the same shape: got shapes %s, %s, %s', image_org.shape, noisy_image.shape, image_reconstructed.shape)
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    filepath = os.path.join(output_dir, f'{image_id}_{round(exp_time)}_{round(new_exp_time)}.png')
    # Pre-initialise optional variables so they are always bound after the try block.
    org_df = rec_df = noisy_df = pd.DataFrame()
    org_a = org_b = org_theta = np.array([])
    rec_a = rec_b = rec_theta = np.array([])

    # Detect sources in original and reconstructed images
    try:
        if kwargs['func'] == wrap_extract_sources:
            x_org, y_org, flux_org, flux_error_org, org_mask, org_a, org_b, org_theta, org_df = kwargs['func'](image_org, 'original', kwargs)
            x_rec, y_rec, flux_rec, flux_error_rec, rec_mask, rec_a, rec_b, rec_theta, rec_df = kwargs['func'](image_reconstructed, 'reconstructed', kwargs)
            x_noisy, y_noisy, flux_noisy, flux_error_noisy, noisy_mask, noisy_a, noisy_b, noisy_theta, noisy_df = kwargs['func'](noisy_image, 'noisy', kwargs)
        else:
            x_org, y_org, flux_org, flux_error_org, org_mask = kwargs['func'](image_org, kwargs)
            x_rec, y_rec, flux_rec, flux_error_rec, rec_mask = kwargs['func'](image_reconstructed, kwargs)
            x_noisy, y_noisy, flux_noisy, flux_error_noisy, noisy_mask = kwargs['func'](noisy_image, kwargs)
    except Exception as err:
        logging.warning(f'could not find light sources: {err}')
        return
    
    # Remove NaNs, keeping original-space index mapping so downstream indexing is consistent
    valid_rec = ~(np.isnan(x_rec) | np.isnan(y_rec))
    orig_idx_rec = np.where(valid_rec)[0]          # maps clean→original index space
    x_rec_clean = x_rec[valid_rec]
    y_rec_clean = y_rec[valid_rec]
    valid_org = ~(np.isnan(x_org) | np.isnan(y_org))
    orig_idx_org = np.where(valid_org)[0]          # maps clean→original index space
    x_org_clean = x_org[valid_org]
    y_org_clean = y_org[valid_org]
    valid_noisy = ~(np.isnan(x_noisy) | np.isnan(y_noisy))
    x_noisy_clean = x_noisy[valid_noisy]
    y_noisy_clean = y_noisy[valid_noisy]

    if kwargs['func'] == wrap_extract_sources:
        org_df = org_df[valid_org].reset_index(drop=True)
        rec_df = rec_df[valid_rec].reset_index(drop=True)
        noisy_df = noisy_df[valid_noisy].reset_index(drop=True)
        org_df['image_id'] = [image_id]*len(org_df)
        rec_df['image_id'] = [image_id]*len(rec_df)
        noisy_df['image_id'] = [image_id]*len(noisy_df)
        org_df['exp_time'] = [exp_time]*len(org_df)
        noisy_df['exp_time'] = [exp_time]*len(noisy_df)
        noisy_df['new_exp_time'] = [new_exp_time]*len(noisy_df)
        rec_df['exp_time'] = [exp_time]*len(rec_df)
        rec_df['new_exp_time'] = [new_exp_time]*len(rec_df)
    else:
        org_df = rec_df = noisy_df = pd.DataFrame()

    # Build KD-tree on clean rec positions; query from clean org positions
    try:
        tree_rec = cKDTree(np.column_stack((x_rec_clean, y_rec_clean)))
        matches = tree_rec.query(np.column_stack((x_org_clean, y_org_clean)), distance_upper_bound=kwargs['distance_threshold'])
        # indices into clean arrays
        matched_clean_org = np.where(matches[0] < kwargs['distance_threshold'])[0]
        matched_clean_rec = matches[1][matched_clean_org]
        # map back to original (unfiltered) index space for consistent flux/plot indexing
        matched_indices_image_org = orig_idx_org[matched_clean_org]
        matched_indices_image_rec = orig_idx_rec[matched_clean_rec]
        unmatched_indices_image_org = np.setdiff1d(np.arange(len(x_org)), matched_indices_image_org)
        unmatched_indices_image_rec = np.setdiff1d(np.arange(len(x_rec)), matched_indices_image_rec)
    except Exception as err:
        logging.warning('compare_images: error finding mutual sources: %s', err)
        return

    mean_ssmi, noisy_psnr, psnr, mean_noisy_ssmi = np.nan, np.nan, np.nan, np.nan
    mse, noisy_mse = np.nan, np.nan
    psnr_L = np.max(image_org) - np.min(image_org)
    try:
        mean_ssmi, ssim = compute_ssim(image_org, image_reconstructed,
                                        kwargs['alpha'], kwargs['beta'],
                                        kwargs['gamma'], kwargs['k1'],
                                        kwargs['k2'], kwargs['win_size'], kwargs['win_sigma'])
        mean_noisy_ssmi, noisy_ssim = compute_ssim(image_org, noisy_image,
                                        kwargs['alpha'], kwargs['beta'],
                                        kwargs['gamma'], kwargs['k1'],
                                        kwargs['k2'], kwargs['win_size'], kwargs['win_sigma'])
        psnr, mse = calculate_psnr(image_org, image_reconstructed)
        noisy_psnr, noisy_mse = calculate_psnr(image_org, noisy_image)
    except Exception as err:
        logging.warning('compare_images: error calculating metrics: %s', err)

    # Metrics Calculation
    tp = len(matched_indices_image_org)
    fp = len(unmatched_indices_image_rec)
    fn = len(unmatched_indices_image_org)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f_measure = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    rec_error_mask = flux_error_rec[matched_indices_image_rec] != 0.0
    org_error_mask = flux_error_org[matched_indices_image_org] != 0.0 
    flux_rec = flux_rec[matched_indices_image_rec][rec_error_mask]  
    flux_error_rec = flux_error_rec[matched_indices_image_rec][rec_error_mask]
    flux_org = flux_org[matched_indices_image_org][org_error_mask]  
    flux_error_org = flux_error_org[matched_indices_image_org][org_error_mask]

    if kwargs['func'] == wrap_extract_sources:
        # org_df/rec_df were already filtered to valid_org/valid_rec rows; map back
        org_df = org_df.iloc[matched_clean_org].reset_index(drop=True)
        rec_df = rec_df.iloc[matched_clean_rec].reset_index(drop=True)

    if isinstance(flux_org, np.ndarray) and isinstance(flux_rec, np.ndarray) and flux_org.shape == flux_rec.shape:
        nonzero_flux = flux_org != 0.0
        rfe = np.mean((flux_org[nonzero_flux] - flux_rec[nonzero_flux]) / flux_org[nonzero_flux]) if nonzero_flux.any() else np.nan
    else:
        rfe = np.nan
    if isinstance(flux_rec, np.ndarray) and isinstance(flux_error_rec, np.ndarray) and flux_rec.shape == flux_error_rec.shape:
        snr_rec = np.mean(flux_rec / flux_error_rec)
    else:
        snr_rec = 0
    if isinstance(flux_org, np.ndarray) and isinstance(flux_error_org, np.ndarray) and flux_org.shape == flux_error_org.shape:
        snr_org = np.mean(flux_org / flux_error_org)
    else:
        snr_org = 0
    
    iou, union = calculate_iou(org_mask, rec_mask)
    stats = {
        'image_id': image_id,
        'org_exp_time': exp_time,
        'new_exp_time': new_exp_time,
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'Precision': precision,
        'Recall': recall,
        'F-measure': f_measure,
        'RFE': rfe,
        'SNR_org': snr_org,
        'SNR_rec': snr_rec,
        'PSNR_rec': psnr,
        'PSNR_noisy': noisy_psnr,
        'MSE_rec': mse if isinstance(mse, float) else np.nan,
        'MSE_noisy': noisy_mse if isinstance(noisy_mse, float) else np.nan,
        'PSNR_L': float(psnr_L),
        'SSIM_rec': mean_ssmi,
        'SSIM_noisy': mean_noisy_ssmi,
        'IoU' : iou,
        'union' : union
    }
    # Plot source comparison and save the image
    try:
        if if_selected:
            if kwargs['func'] == wrap_extract_sources:
                plot_source_comparison_sep(image_org, noisy_image, image_reconstructed, 
                                x_org, y_org, x_rec, y_rec, 
                                matched_indices_image_org, unmatched_indices_image_org, 
                                matched_indices_image_rec, unmatched_indices_image_rec,
                                org_a, org_b, rec_a, rec_b, org_theta, rec_theta,
                                filepath)
            else:
                plot_source_comparison(image_org, noisy_image, image_reconstructed, 
                                x_org, y_org, x_rec, y_rec, 
                                matched_indices_image_org, unmatched_indices_image_org, 
                                matched_indices_image_rec, unmatched_indices_image_rec,
                                filepath) 
    except Exception as err:
        logging.warning('compare_images: error saving source comparison plot: %s', err)
    
    return stats, (flux_rec.flatten().tolist(),
                   flux_org.flatten().tolist()
                   ),(flux_error_rec.flatten().tolist(),
                  flux_error_org.flatten().tolist()
                  ), (org_df, noisy_df, rec_df)

def decide_scale(model_dir):
    """Infer the scaling/normalisation function pair from checkpoint info or path.

    Inspects the directory name for known substrings and returns the matching
    forward and inverse transform callables.

    Parameters
    ----------
    model_dir : str
        Path or name of the model directory.

    Returns
    -------
    tuple or None
        ``(scale_fn, descale_fn)`` where both are callables, or ``None`` if no
        known scaling keyword is found in *model_dir*.
    """
    checkpoint_info = read_checkpoint_info(model_dir) if os.path.isfile(model_dir) else {}
    scaling = checkpoint_info.get('scaling') if isinstance(checkpoint_info, dict) else None
    if scaling == 'log_min_max' or 'log_min_max' in model_dir:
        return (adaptive_log_transform_and_normalize, inverse_adaptive_log_transform_and_denormalize)
    elif scaling == 'min_max' or 'min_max' in model_dir:
        return (min_max_normalization, inverse_min_max_normalization)
    elif scaling == 'z_scale' or 'z_scale' in model_dir:
        return (zscore_normalization, inverse_zscore_normalization)
    else:
        return None

def find_best_performing_models(models_dir, condition, filter_model, model_prototype, n, index, concurrent_workers):
    """Walk model folders, parse template tags, and select checkpoints.

    Parameters
    ----------
    models_dir : str
        Root directory containing hierarchical model directories.
    condition : callable
        Predicate used to keep/discard a discovered model directory.
        Preferred signature is ``condition(model_info: dict) -> bool`` where
        ``model_info`` contains keys parsed from the directory template:
        ``is_gan``, ``use_attention``, ``loss_function``, ``data_alias_enriched_hex``,
        ``scaling_tag``, ``dropout_tag``, ``activation_tag``,
        ``output_activation_tag``, ``discriminator_activation_tag``,
        ``discriminator_output_activation_tag``, plus ``model_dir`` and
        ``checkpoints_dir``.
        Backward-compatible signature ``condition(model_dir: str)`` is also
        supported.
    filter_model : callable
        Function that filters/selects checkpoint files from one checkpoints
        directory.
    model_prototype : str
        Glob pattern (relative to each model sub-directory) used to discover
        checkpoint files, e.g. ``'*.keras'``.
    n : int
        Maximum number of checkpoints to select per model.
    index : int
        Index used to select a specific model after shuffling.
    concurrent_workers : int
        Number of concurrent workers used for model selection.

    Returns
    -------
    dict
        Mapping from model directory (relative to models root) to selected
        checkpoint file paths.
    """
    template_keys = [
        'is_gan',
        'use_attention',
        'loss_function',
        'data_alias_enriched_hex',
        'scaling_tag',
        'dropout_tag',
        'activation_tag',
        'output_activation_tag',
        'discriminator_activation_tag',
        'discriminator_output_activation_tag',
    ]

    def _extract_epoch(model_file):
        """Extract epoch integer from callback naming (model_XXXX / best_model_XXXX)."""
        base = os.path.basename(model_file).split('.')[0]
        if base == 'final_model':
            return -1
        for part in reversed(base.split('_')):
            try:
                return int(part)
            except ValueError:
                continue
        return None

    def _apply_filter_model(file_list):
        """Apply filter_model with backward-compatible call signatures."""
        try:
            return filter_model(file_list, model_prototype, n)
        except TypeError:
            try:
                return filter_model(file_list, n)
            except TypeError:
                try:
                    return filter_model(file_list, model_prototype)
                except TypeError:
                    return filter_model(file_list)

    # Use ./models as search root, even if a deeper path was passed.
    search_root = os.path.abspath(models_dir)
    while os.path.basename(search_root).lower() != 'models' and os.path.dirname(search_root) != search_root:
        search_root = os.path.dirname(search_root)

    if not isinstance(model_prototype, str) or not model_prototype:
        raise ValueError('model_prototype must be a non-empty string.')
    checkpoint_pattern = model_prototype
    discovered_models = {}

    for root, _dirs, _files in os.walk(search_root):
        if os.path.basename(root) != 'checkpoints':
            continue

        model_dir_path = os.path.dirname(root)
        rel_model_dir = os.path.relpath(model_dir_path, search_root)
        rel_parts = rel_model_dir.split(os.sep)

        # Remove checkpoint folders that do not follow the exact template shape.
        if len(rel_parts) != len(template_keys):
            continue

        checkpoint_files = sorted(glob.glob(os.path.join(root, checkpoint_pattern)))
        if not checkpoint_files:
            continue

        epochs_by_file = {
            checkpoint_file: _extract_epoch(checkpoint_file)
            for checkpoint_file in checkpoint_files
        }

        selected_models = _apply_filter_model(checkpoint_files)
        selected_models = [model_file for model_file in selected_models if model_file in epochs_by_file]
        if not selected_models:
            continue

        base_info = dict(zip(template_keys, rel_parts))
        base_info['model_dir'] = rel_model_dir
        base_info['checkpoints_dir'] = root

        rows = []
        for checkpoint_file in selected_models:
            row = dict(base_info)
            row['filepath'] = checkpoint_file
            basename = os.path.basename(checkpoint_file)
            row['prefix'] = 'best_model' if basename.startswith('best_model') else 'model'
            row['epoch'] = epochs_by_file[checkpoint_file]
            rows.append(row)

        model_info = pd.DataFrame(rows)
        # best_model rows first, then descending epoch within each prefix group
        model_info['_prefix_order'] = (model_info['prefix'] != 'best_model').astype(int)
        model_info = model_info.sort_values(
            ['_prefix_order', 'epoch'], ascending=[True, False]
        ).drop(columns=['_prefix_order']).reset_index(drop=True)

        try:
            mask = condition(model_info)
            filtered_info = model_info.loc[mask]
        except Exception:
            try:
                include_model = bool(condition(rel_model_dir))
                filtered_info = model_info if include_model else pd.DataFrame()
            except Exception:
                filtered_info = model_info

        if filtered_info.empty:
            continue

        selected_models = filtered_info['filepath'].tolist()
        discovered_models[rel_model_dir] = selected_models
    discovered_model_keys = list(discovered_models.keys())
    random.seed(42)  # Set a fixed seed for reproducibility
    random.shuffle(discovered_model_keys)

    n_model_dirs = int(np.ceil(len(discovered_model_keys) / concurrent_workers))
    start_index = index * n_model_dirs
    end_index = min(start_index + n_model_dirs, len(discovered_model_keys))
    discovered_models = {key: discovered_models[key] for key in discovered_model_keys[start_index:end_index]}
    logging.info('find_best_performing_models: selected %d model directories', len(discovered_models))
    return discovered_models

def get_test_images(images, kwargs_data, scaling):
    """Build the test-image DataFrame by running the data augmentation pipeline.

    Calls :func:`~src.training.new_train.data_augment_pluggable` with
    ``test=True`` to obtain the held-out subset metadata.

    Parameters
    ----------
    images : list
        List of image paths or objects passed to the augmentation pipeline.
    kwargs_data : dict
        Keyword arguments forwarded to
        :func:`~src.training.new_train.data_augment_pluggable`. The key
        ``'test'`` is set to ``True`` inside this function.
    scaling : object
        Scaling specification forwarded to
        :func:`~src.training.new_train.data_augment_pluggable`.

    Returns
    -------
    pandas.DataFrame
        Test-set metadata DataFrame with at minimum a file-path column and an
        exposure-time column.
    """
    kwargs_data['test'] = True
    return data_augment_pluggable(images, kwargs_data, scaling)

def _reconstruct_patch(noisy_patch, scales, model, use_mosaic, patch_size, stride, weighting, batch_size, gaussian_sigma=64):
    """Reconstruct a single patch using mosaic sliding-window or direct model inference.

    Parameters
    ----------
    noisy_patch : numpy.ndarray
        2-D noisy image patch to reconstruct.
    scales : tuple or None
        ``(scale_fn, descale_fn)`` callables, or ``None`` for no scaling.
    model : tf.keras.Model
        Trained model.
    use_mosaic : bool
        If ``True``, use :func:`sliding_window_inference`; otherwise call
        ``model.predict`` directly on the whole patch.
    patch_size : tuple of int
        Sliding-window patch size ``(H, W, C)``.
    stride : tuple of int
        Sliding-window stride ``(H, W, C)``.
    weighting : str
        Overlap-blending strategy passed to :func:`sliding_window_inference`.
    batch_size : int
        Patches per model call.
    gaussian_sigma : float, optional
        Gaussian weighting sigma. Default is 64.

    Returns
    -------
    numpy.ndarray or None
        Reconstructed 2-D image array, or ``None`` on failure.
    """
    reconstructed_image = None
    if scales is not None:
        scale, descale = scales
        try:
            args = scale(noisy_patch)
            if args is None:
                return None
            scaled_image = args[0]
            func_args = args[1:]
            try:
                if use_mosaic:
                    out = sliding_window_inference(np.expand_dims(scaled_image, axis=-1), model,
                                                  patch_size, stride, weighting, batch_size,
                                                  gaussian_sigma)
                    if out is not None:
                        reconstructed_image = descale(out[:, :, 0], *func_args)
                else:
                    out = model.predict(np.array([np.expand_dims(scaled_image, axis=-1)]))
                    reconstructed_image = descale(out[0, :, :, 0], *func_args)
            except Exception as err:
                logging.warning('_reconstruct_patch: error during reconstruction: %s', err)
                return None
        except Exception as e:
            logging.warning('_reconstruct_patch: error in scaling: %s', e)
            return None
    else:
        try:
            if use_mosaic:
                out = sliding_window_inference(np.expand_dims(noisy_patch, axis=-1), model,
                                               patch_size, stride, weighting, batch_size,
                                               gaussian_sigma)
                if out is not None:
                    reconstructed_image = out[:, :, 0]
            else:
                out = model.predict(np.array([np.expand_dims(noisy_patch, axis=-1)]))
                reconstructed_image = out[0, :, :, 0]
        except Exception as err:
            logging.warning('_reconstruct_patch: error during reconstruction: %s', err)
            return None
    return reconstructed_image

def _collect_results(results, all_flux_rec, all_flux_org,
                     all_flux_error_rec, all_flux_error_org, metrics, org_dfs, noisy_dfs, rec_dfs, model_index):
    """Unpack :func:`compare_images` results and append to accumulator lists.

    Parameters
    ----------
    results : tuple
        Return value of :func:`compare_images`:
        ``(stats, (flux_rec, flux_org), (flux_error_rec, flux_error_org),
        (org_df, noisy_df, rec_df))``.
    all_flux_rec, all_flux_org : list
        Accumulators for matched reconstructed and original flux values.
    all_flux_error_rec, all_flux_error_org : list
        Accumulators for matched flux uncertainties.
    metrics : list
        Accumulator for per-image statistics dictionaries.
    org_dfs, noisy_dfs, rec_dfs : list
        Per-image noisy-source catalogues (only populated for
        ``model_index == 0``).
    model_index : int
        Index of the current model. When non-zero, *org_dfs* and *noisy_dfs*
        are returned as empty DataFrames.

    Returns
    -------
    None
        The function mutates the passed accumulator lists in-place.
    """
    stats, (flux_rec, flux_org), (flux_error_rec, flux_error_org), (org_df, noisy_df, rec_df) = results

    if stats is not None:
        metrics.append(stats)

    if flux_rec is not None:
        all_flux_rec.extend(flux_rec)
    if flux_org is not None:
        all_flux_org.extend(flux_org)
    if flux_error_rec is not None:
        all_flux_error_rec.extend(flux_error_rec)
    if flux_error_org is not None:
        all_flux_error_org.extend(flux_error_org)

    if isinstance(rec_df, pd.DataFrame) and not rec_df.empty:
        rec_dfs.append(rec_df)
    if model_index == 0:
        if isinstance(org_df, pd.DataFrame) and not org_df.empty:
            org_dfs.append(org_df)
        if isinstance(noisy_df, pd.DataFrame) and not noisy_df.empty:
            noisy_dfs.append(noisy_df)


def _finalise_dfs(rec_dfs, org_dfs, noisy_dfs, epoch, model_dir, model_index):
    """Concatenate per-image source DataFrames and attach model metadata.

    Parameters
    ----------
    rec_dfs, org_dfs, noisy_dfs : list of pandas.DataFrame
        Per-image catalog DataFrames collected during evaluation.
    epoch : str
        Epoch identifier to append to output DataFrames.
    model_dir : str
        Model directory label to append to output DataFrames.
    model_index : int
        Model index; when non-zero, original/noisy outputs are suppressed.

    Returns
    -------
    tuple
        ``(org_df, noisy_df, rec_df)`` concatenated DataFrames.
    """
    if len(rec_dfs):
        if model_index == 0:
            org_df = pd.concat(org_dfs, ignore_index=True) if len(org_dfs) else pd.DataFrame()
            noisy_df = pd.concat(noisy_dfs, ignore_index=True) if len(noisy_dfs) else pd.DataFrame()
            if not org_df.empty:
                org_df['epoch'] = epoch
                org_df['model'] = model_dir
            if not noisy_df.empty:
                noisy_df['epoch'] = epoch
                noisy_df['model'] = model_dir
        else:
            org_df = pd.DataFrame()
            noisy_df = pd.DataFrame()

        rec_df = pd.concat(rec_dfs, ignore_index=True)
        rec_df['epoch'] = epoch
        rec_df['model'] = model_dir
    else:
        org_df = pd.DataFrame()
        noisy_df = pd.DataFrame()
        rec_df = pd.DataFrame()
    return org_df, noisy_df, rec_df

def process_single_model(model_file, model_dir, scales, test_images_df, kwargs_source,
                         patch_size=(256, 256, 1), stride=(128, 128, 1), weighting='gaussian', batch_size=16,
                         frac=0.1, model_index=0, use_mosaic=True,
                         gaussian_sigma=64,
                         type_of_image='SCI',
                         nan_value=0.0, posinf_value=0.0, neginf_value=0.0,
                         location_col='location', exp_time_col='sci_actual_duration', new_exp_time_col='new_exp_time', sigma_key='combined_sigma',
                         noise_fn=_simulated_image_from_exposure,
                         combined_images_dir='./metrics_updated/combined_images', png_dir='./metrics_updated/pngs',
                         org_dir='./metrics_updated/original_images', noisy_dir='./metrics_updated/noisy_images',
                         rec_dir='./metrics_updated/reconstructed_images', model_alias_hex=''):
    """Evaluate one model checkpoint over all rows in *test_images_df*.

    Parameters
    ----------
    model_file : str
        Path to the Keras checkpoint file to load.
    model_dir : str
        Parent model directory name (used for labelling output rows).
    scales : tuple or None
        ``(scale_fn, descale_fn)`` from :func:`decide_scale`, or ``None``.
    test_images_df : pandas.DataFrame
        Metadata DataFrame for the test set.
    kwargs_source : dict
        Source-detection and metrics parameters (see :func:`compare_images`).
    patch_size : tuple of int, optional
        Sliding-window patch size ``(H, W, C)``. Default is
        ``(256, 256, 1)``.
    stride : tuple of int, optional
        Sliding-window stride ``(H, W, C)``. Default is ``(128, 128, 1)``.
    weighting : str, optional
        Overlap-blending strategy. Default is ``'gaussian'``.
    batch_size : int, optional
        Patches per model call. Default is 16.
    frac : float, optional
        Fraction of images for which comparison plots are saved. Default is
        0.1.
    model_index : int, optional
        Index of the model in the evaluation run; original/noisy catalogues
        are only written when this is 0. Default is 0.
    use_mosaic : bool, optional
        If ``True``, reconstruct via sliding-window mosaic; if ``False``,
        reconstruct via a single ``model.predict`` call per crop. Default is
        ``True``.
    gaussian_sigma : float, optional
        Gaussian weighting sigma. Default is 64.
    nan_value, posinf_value, neginf_value : float, optional
        Replacement values for NaN/+Inf/-Inf pixels. Defaults are 0.0.
    location_col : str, optional
        Column name for image file paths. Default is ``'location'``.
    exp_time_col : str, optional
        Column name for original exposure time. Default is
        ``'sci_actual_duration'``.
    new_exp_time_col : str, optional
        Column name for simulated exposure time. Default is
        ``'new_exp_time'``.
    sigma_key : str, optional
        Column name for noise sigma. Default is ``'combined_sigma'``.
    noise_fn : callable, optional
        Function that generates a noisy image from the original. Default is
        :func:`~src.training.math_helpers._simulated_image_from_exposure`.
    combined_images_dir, png_dir, org_dir, noisy_dir, rec_dir : str, optional
        Output directory paths for different artefact types.

    Returns
    -------
    metrics : pandas.DataFrame
        Per-image metrics.
    aggregated_metrics : pandas.DataFrame
        Single-row aggregated metrics.
    dfs : tuple
        ``(org_dfs, noisy_dfs, rec_dfs)`` — source catalogues as DataFrames.
    """
    epoch = os.path.basename(model_file).split('.')[0].split('_')[-1]
    required_columns = [location_col, exp_time_col, new_exp_time_col, sigma_key]
    if not all(col in test_images_df.columns for col in required_columns):
        logging.warning('process_single_model: DataFrame missing required columns: %s', required_columns)
        return
    try:
        model = load_checkpoint_model(
            model_file,
            compile=False,
            custom_objects=build_checkpoint_custom_objects(),
        )
    except Exception as err:
        logging.error('process_single_model: failed to load model %s: %s', model_file, err)
        return

    combined_images_dir = combined_images_dir.replace('*', model_alias_hex)
    png_dir = png_dir.replace('*', model_alias_hex)
    org_dir = org_dir.replace('*', model_alias_hex)
    noisy_dir = noisy_dir.replace('*', model_alias_hex)
    rec_dir = rec_dir.replace('*', model_alias_hex)

    metrics = []
    aggregated_metrics = []
    all_flux_rec, all_flux_org = [], []
    all_flux_error_rec, all_flux_error_org = [], []
    org_dfs, noisy_dfs, rec_dfs = [], [], []

    for _, row in test_images_df.iterrows():
        image_filepath = row[location_col]
        org_exp_time = row[exp_time_col]
        new_exp_time = row[new_exp_time_col]
        new_sigma = row[sigma_key]

        try:
            image = open_fits(image_filepath, type_of_image=type_of_image)
        except Exception as e:
            logging.warning('process_single_model: error reading image %s: %s', image_filepath, e)
            continue
        if image is None:
            continue
        if not isinstance(image, np.ndarray):
            logging.warning('process_single_model: open_fits returned unexpected type %s for %s', type(image), image_filepath)
            continue

        image = np.nan_to_num(image, nan=nan_value, 
                              posinf=posinf_value, 
                              neginf=neginf_value)
        org_name = os.path.basename(image_filepath).rsplit('.', 1)[0]

        # Mosaic mode: whole image at once; no-mosaic mode: iterate over crops
        patches = [(image, org_name, 0)] if use_mosaic else [
            (np.nan_to_num(crop, nan=nan_value, 
                           posinf=posinf_value, 
                           neginf=neginf_value), f'{org_name}_{i}', i)
            for i, crop in enumerate(crop_image_generator(image, ps=patch_size[0]))
        ]

        for base_image, image_id, crop_idx in patches:
            if_selected = np.random.rand() <= frac
            noisy_image = noise_fn(base_image, row, new_sigma)
            noisy_image = np.nan_to_num(noisy_image, nan=nan_value, 
                                        posinf=posinf_value, 
                                        neginf=neginf_value)

            reconstructed_image = _reconstruct_patch(
                noisy_image, scales, model, use_mosaic,
                patch_size, stride, weighting, batch_size,
                gaussian_sigma,
            )
            if reconstructed_image is None:
                logging.warning('process_single_model: reconstruction failed for %s, skipping', image_id)
                continue
            reconstructed_image = np.nan_to_num(reconstructed_image, nan=nan_value, posinf=posinf_value, neginf=neginf_value)

            # Create output directories
            combined_image_dir = os.path.join(combined_images_dir, str(epoch))
            try:
                for folder in [combined_image_dir, png_dir, org_dir, noisy_dir, rec_dir]:
                    ensure_directory_exists(folder)
            except Exception as e:
                logging.warning('process_single_model: error creating output directories: %s', e)
                continue

            # Save FITS and PNG images
            for index, (img, suffix) in enumerate([(base_image, "_org"),
                                                    (reconstructed_image, "_rec"),
                                                    (noisy_image, "_noisy")]):
                try:
                    if index == 0:
                        filename = f"{org_name}{suffix}_{round(org_exp_time)}.fits"
                        if if_selected:
                            save_fits(img, filename, org_dir, type_of_image=type_of_image)
                    elif index == 1:
                        filename = f"{org_name}{suffix}_{epoch}_{round(org_exp_time)}_{round(new_exp_time)}.fits"
                        if if_selected:
                            save_fits(img, filename, rec_dir, type_of_image=type_of_image)
                    else:
                        filename = f"{org_name}{suffix}_{round(org_exp_time)}_{round(new_exp_time)}.fits"
                        if if_selected:
                            save_fits(img, filename, noisy_dir, type_of_image=type_of_image)
                    try:
                        scaled_img, vmin, vmax = scale_image(img)
                        scaled_img.save(os.path.join(png_dir, f"{filename.split('.')[0]}.png"))
                    except Exception as e:
                        logging.warning('process_single_model: error scaling/saving PNG for %s%s: %s', org_name, suffix, e)
                except Exception as e:
                    logging.warning('process_single_model: error saving images for %s%s: %s', org_name, suffix, e)
                    continue

            results = compare_images(base_image, noisy_image, reconstructed_image, image_id,
                                     float(org_exp_time), float(new_exp_time),
                                     combined_image_dir, kwargs_source, if_selected)
            if results is not None:
                _collect_results(results,
                                 all_flux_rec, all_flux_org, all_flux_error_rec, all_flux_error_org,
                                 metrics, org_dfs, noisy_dfs, rec_dfs, model_index)

    metrics = pd.DataFrame(metrics)
    all_flux_rec = np.array(all_flux_rec)
    all_flux_org = np.array(all_flux_org)
    all_flux_error_rec = np.array(all_flux_error_rec)
    all_flux_error_org = np.array(all_flux_error_org)

    org_dfs, noisy_dfs, rec_dfs = _finalise_dfs(rec_dfs, org_dfs, noisy_dfs, epoch, model_dir, model_index)

    try:
        aggregated_stats = aggregate_df(metrics, all_flux_rec, all_flux_org, all_flux_error_rec, all_flux_error_org)
        if aggregated_stats is not None:
            aggregated_stats['epoch'] = epoch
            aggregated_stats['model'] = model_dir
            aggregated_metrics.append(aggregated_stats)
    except Exception as err:
        logging.warning('process_single_model: aggregate_df failed: %s', err)
    return metrics, pd.DataFrame(aggregated_metrics), (org_dfs, noisy_dfs, rec_dfs)


def process_models(job, kwargs_source,
                    workers=8, frac=0.1, parallel=True, patch_size=(256, 256, 1), stride=(128, 128, 1),
                    weighting='gaussian', batch_size=16, use_mosaic=True,
                    gaussian_sigma=64,
                    type_of_image='SCI',
                    nan_value=0.0, posinf_value=0.0, neginf_value=0.0,
                    location_col='location', exp_time_col='sci_actual_duration', new_exp_time_col='new_exp_time', sigma_key='combined_sigma',
                    noise_fn=_simulated_image_from_exposure,
                    combined_images_dir='./metrics_updated/combined_images', png_dir='./metrics_updated/pngs',
                    org_dir='./metrics_updated/original_images', noisy_dir='./metrics_updated/noisy_images',
                    rec_dir='./metrics_updated/reconstructed_images',
                    all_metrics_csv=None, aggregated_metrics_csv=None,
                    org_catalog_csv=None, noisy_catalog_csv=None, rec_catalog_csv=None):
    """Evaluate all checkpoints from a single model directory.

    Dispatches to :func:`process_single_model` — optionally in parallel —
    for each checkpoint file found in the job tuple, then concatenates and
    returns the collected results.

    Parameters
    ----------
    job : tuple
        ``(model_dir, model_files, scales, test_images_df)`` as returned by
        :func:`find_best_performing_models` and :func:`decide_scale`.
    kwargs_source : dict
        Source-detection and metrics parameters forwarded to each
        :func:`process_single_model` call.
    workers : int, optional
        Number of parallel worker processes. Default is 8.
    frac : float, optional
        Fraction of images for which comparison plots are saved. Default is
        0.1.
    parallel : bool, optional
        If ``True``, use a ``ProcessPoolExecutor``; otherwise run
        sequentially. Default is ``True``.
    patch_size : tuple of int, optional
        Sliding-window patch size ``(H, W, C)``. Default is
        ``(256, 256, 1)``.
    stride : tuple of int, optional
        Sliding-window stride. Default is ``(128, 128, 1)``.
    weighting : str, optional
        Overlap-blending strategy. Default is ``'gaussian'``.
    batch_size : int, optional
        Patches per model call. Default is 16.
    use_mosaic : bool, optional
        Whether to use sliding-window mosaic inference. Default is ``True``.
    gaussian_sigma : float, optional
        Gaussian weighting sigma. Default is 64.
    nan_value, posinf_value, neginf_value : float, optional
        Replacement values for non-finite pixels. Defaults are 0.0.
    location_col, exp_time_col, new_exp_time_col, sigma_key : str, optional
        Column names in the test-images DataFrame.
    noise_fn : callable, optional
        Noise-simulation function.
    combined_images_dir, png_dir, org_dir, noisy_dir, rec_dir : str, optional
        Output directory paths.

    Returns
    -------
    all_metrics : pandas.DataFrame
        Concatenated per-image metrics across all checkpoints.
    aggregated_metrics : pandas.DataFrame
        Concatenated aggregated metrics across all checkpoints.
    dfs : tuple
        ``(org_dfs, noisy_dfs, rec_dfs)`` — concatenated source catalogues.
    """

    if job is None or len(job) != 5:
        logging.warning('process_models: invalid job tuple (expected 5-element tuple, got %s)', job)
        return pd.DataFrame(), pd.DataFrame(), (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    model_dir, model_files, scales, test_images_df, model_tags = job
    
    def _require_str_path(path_value, name):
        if not isinstance(path_value, str):
            raise ValueError(f'process_models requires "{name}" as a non-empty string template.')
        return path_value

    all_metrics_csv_s = _require_str_path(all_metrics_csv, 'all_metrics_csv')
    aggregated_metrics_csv_s = _require_str_path(aggregated_metrics_csv, 'aggregated_metrics_csv')
    org_catalog_csv_s = _require_str_path(org_catalog_csv, 'org_catalog_csv')
    noisy_catalog_csv_s = _require_str_path(noisy_catalog_csv, 'noisy_catalog_csv')
    rec_catalog_csv_s = _require_str_path(rec_catalog_csv, 'rec_catalog_csv')

    all_metrics_csv = all_metrics_csv_s.replace('*', model_tags['model_alias_hex'])
    aggregated_metrics_csv = aggregated_metrics_csv_s.replace('*', model_tags['model_alias_hex'])
    org_catalog_csv = org_catalog_csv_s.replace('*', model_tags['model_alias_hex'])
    noisy_catalog_csv = noisy_catalog_csv_s.replace('*', model_tags['model_alias_hex'])
    rec_catalog_csv = rec_catalog_csv_s.replace('*', model_tags['model_alias_hex'])

    _acc_metrics: list = []
    _acc_aggregated: list = []
    _acc_org_dfs: list = []
    _acc_noisy_dfs: list = []
    _acc_rec_dfs: list = []

    def _collect_model_result(result):
        if result is None:
            return
        metrics, aggregated_metric, (org_df, noisy_df, rec_df) = result
        if not metrics.empty:
            _acc_metrics.append(metrics)
        if not aggregated_metric.empty:
            _acc_aggregated.append(aggregated_metric)
        if not org_df.empty:
            _acc_org_dfs.append(org_df)
            _acc_noisy_dfs.append(noisy_df)
        if not rec_df.empty:
            _acc_rec_dfs.append(rec_df)

    if parallel:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_single_model, model_file, model_dir, scales,
                                       test_images_df, kwargs_source, patch_size, stride,
                                       weighting, batch_size, frac, index, use_mosaic,
                                       gaussian_sigma,
                                       type_of_image,
                                       nan_value, posinf_value,
                                       neginf_value,
                                       location_col, exp_time_col, new_exp_time_col, sigma_key,
                                       noise_fn,
                                       combined_images_dir, png_dir, org_dir, noisy_dir, rec_dir,
                                       model_tags['model_alias_hex'])
                       for index, model_file in enumerate(model_files)]
        for future in as_completed(futures):
            if future is not None:
                try:
                    _collect_model_result(future.result())
                except Exception as err:
                    logging.warning('process_models: error processing model future: %s', err)
    else:
        for index, model_file in enumerate(model_files):
            _collect_model_result(process_single_model(model_file, model_dir, scales, test_images_df,
                                                       kwargs_source, patch_size, stride,
                                                       weighting, batch_size, frac, index, use_mosaic,
                                                       gaussian_sigma,
                                                       type_of_image,
                                                       nan_value, posinf_value,
                                                       neginf_value,
                                                       location_col, exp_time_col, new_exp_time_col, sigma_key,
                                                       noise_fn,
                                                       combined_images_dir, png_dir, org_dir, noisy_dir, rec_dir,
                                                       model_tags['model_alias_hex']))

    if _acc_metrics:
        all_metrics = pd.concat(_acc_metrics, ignore_index=True)
    else:
        all_metrics = pd.DataFrame()
    if _acc_aggregated:
        aggregated_metrics = pd.concat(_acc_aggregated, ignore_index=True)
    else:
        aggregated_metrics = pd.DataFrame()

    if len(_acc_rec_dfs):
        org_dfs = pd.concat(_acc_org_dfs, ignore_index=True) if _acc_org_dfs else pd.DataFrame()
        noisy_dfs = pd.concat(_acc_noisy_dfs, ignore_index=True) if _acc_noisy_dfs else pd.DataFrame()
        rec_dfs = pd.concat(_acc_rec_dfs, ignore_index=True)
    else:
        org_dfs = pd.DataFrame()
        noisy_dfs = pd.DataFrame()
        rec_dfs = pd.DataFrame()

    if not all_metrics.empty and all_metrics_csv:
        ensure_parent_dir_exists(all_metrics_csv)
        all_metrics.to_csv(all_metrics_csv, index=False)
    if not aggregated_metrics.empty and aggregated_metrics_csv:
        ensure_parent_dir_exists(aggregated_metrics_csv)
        aggregated_metrics.to_csv(aggregated_metrics_csv, index=False)
    if not org_dfs.empty:
        if org_catalog_csv:
            ensure_parent_dir_exists(org_catalog_csv)
            org_dfs.to_csv(org_catalog_csv, index=False)
        if noisy_catalog_csv:
            ensure_parent_dir_exists(noisy_catalog_csv)
            noisy_dfs.to_csv(noisy_catalog_csv, index=False)
    if not rec_dfs.empty and rec_catalog_csv:
        ensure_parent_dir_exists(rec_catalog_csv)
        rec_dfs.to_csv(rec_catalog_csv, index=False)
    return all_metrics, aggregated_metrics, (org_dfs, noisy_dfs, rec_dfs)
        
def main(models_dir, data_kwargs, model_kwargs, kwargs_source, total_workers, max_workers, frac, condition, filter_model, n, model_prototype,
         all_metrics_csv='./metrics_updated/all_metrics_*.csv',
         aggregated_metrics_csv='./metrics_updated/aggregated_metrics_*.csv',
         org_catalog_csv='./metrics_updated/org_catalog_*.csv',
         noisy_catalog_csv='./metrics_updated/noisy_catalog_*.csv',
         rec_catalog_csv='./metrics_updated/rec_catalog_*.csv',
         parallel=True, parallel_epoch=False, scaling=None, 
         index=0, concurrent_workers=1):
    """Run the full evaluation pipeline across all qualifying model directories.

    Discovers models with :func:`find_best_performing_models`, evaluates each
    with :func:`process_models` (optionally in parallel), and writes per-image
    and aggregated metrics together with source catalogues to CSV files.

    Parameters
    ----------
    models_dir : str
        Root directory containing model sub-directories.
    data_kwargs : dict
        Keyword arguments for :func:`get_test_images`.
    model_kwargs : dict
        Keyword arguments forwarded to :func:`process_models`.
    kwargs_source : dict
        Source-detection and metrics parameters.
    total_workers : int
        Total number of worker processes available.
    max_workers : int
        Maximum number of concurrent model-directory jobs.
    frac : float
        Fraction of images for which comparison plots are saved.
    condition : callable
        ``condition(subdir_name) -> bool`` — filter for model sub-directories.
    filter_model : callable
        Checkpoint selection function (e.g. :func:`get_model_by_modulo`).
    n : int
        Maximum checkpoints per model directory.
    model_prototype : str
        Glob pattern for checkpoint files.
    all_metrics_csv : str, optional
        Output path for the per-image metrics CSV.
    aggregated_metrics_csv : str, optional
        Output path for the aggregated metrics CSV.
    org_catalog_csv : str, optional
        Output path for the original-source catalogue CSV.
    noisy_catalog_csv : str, optional
        Output path for the noisy-source catalogue CSV.
    rec_catalog_csv : str, optional
        Output path for the reconstructed-source catalogue CSV.
    parallel : bool, optional
        If ``True``, process model directories in parallel. Default is
        ``True``.
    parallel_epoch : bool, optional
        If ``True``, process checkpoints within each model directory in
        parallel. Default is ``False``.
    scaling : str, optional
        Scaling method to apply to the test images. Default is ``None``.
    index : int, optional
        Index of the current worker. Default is ``0``.
    concurrent_workers : int, optional
        Number of concurrent workers used for model selection. Default is ``1``.
    """
    try:

        test_images_df = get_test_images(None, data_kwargs["kwargs_data"], scaling=scaling)
    except Exception as err:
        logging.warning(f"problem occurred while getting test images: {err}")
        return 
    if test_images_df is None:
        logging.warning("get_test_images returned None, aborting.")
        return 

    ensure_parent_dir_exists(all_metrics_csv)

    my_dict = find_best_performing_models(models_dir, condition, filter_model, model_prototype, n, 
                                          index, concurrent_workers)
    logging.info('main: model dict: %s', my_dict)

    jobs = []
    for model_dir in my_dict.keys():
        scales = decide_scale(model_dir)
        jobs.append((model_dir, my_dict[model_dir], scales, test_images_df, 
                     _decode_models_dir(os.path.dirname(model_dir))))

    if parallel:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_models,
                                        job,
                                        kwargs_source,
                                        total_workers//max_workers, frac, parallel_epoch,
                                        **model_kwargs,
                                        all_metrics_csv=all_metrics_csv,
                                        aggregated_metrics_csv=aggregated_metrics_csv,
                                        org_catalog_csv=org_catalog_csv,
                                        noisy_catalog_csv=noisy_catalog_csv,
                                        rec_catalog_csv=rec_catalog_csv) for job in jobs]
        for future in as_completed(futures):
            if future is None:
                continue
            try:
                future.result()
            except Exception as err:
                logging.warning('main: error processing job future: %s', err)
    else:
        for job in jobs:
            try:
                process_models(job, kwargs_source,
                               workers=total_workers//max_workers, frac=frac,
                               parallel=parallel_epoch, **model_kwargs,
                               all_metrics_csv=all_metrics_csv,
                               aggregated_metrics_csv=aggregated_metrics_csv,
                               org_catalog_csv=org_catalog_csv,
                               noisy_catalog_csv=noisy_catalog_csv,
                               rec_catalog_csv=rec_catalog_csv)
            except Exception as err:
                logging.warning('main: error processing job: %s', err)

def condition(model_info):
    """Default checkpoint filter — keeps all checkpoints.

    Receives the full *model_info* DataFrame for one model directory and
    returns an index or boolean mask used to select rows (checkpoints).
    Each row carries:

    ``is_gan``, ``use_attention``, ``loss_function``,
    ``data_alias_enriched_hex``, ``scaling_tag``, ``dropout_tag``,
    ``activation_tag``, ``output_activation_tag``,
    ``discriminator_activation_tag``, ``discriminator_output_activation_tag``,
    ``model_dir``, ``checkpoints_dir``, ``filepath``, ``prefix``, ``epoch``.

    Rows are pre-sorted: ``best_model`` prefix first, then descending epoch.

    The fallback signature ``condition(model_dir: str) -> bool`` is also
    supported for backward compatibility (keep/drop the whole directory).

    Parameters
    ----------
    model_info : pandas.DataFrame
        One model directory's checkpoint DataFrame.

    Returns
    -------
    index-like
        Any value accepted by ``DataFrame.loc[]``: a boolean Series, an
        integer array/list of positional labels, or a slice.
    """
    return len(model_info) * [True]

def get_top_best_models(file_list, x):
    """Select the *x* most recent checkpoints by epoch number.

    Epoch numbers are parsed from the last underscore-separated numeric token
    in each filename.  Files with non-parseable names are silently skipped.

    Parameters
    ----------
    file_list : list of str
        Candidate checkpoint file paths.
    x : int
        Number of top checkpoints to return.

    Returns
    -------
    list of str
        Up to *x* file paths sorted by descending epoch number.
    """
    if not isinstance(x, int) or x <= 0:
        return []

    valid_files = []
    for f in file_list:
        try:
            base = os.path.basename(f)
            num_part = base.split('_')[-1].split('.')[0]
            num = int(num_part)
            valid_files.append((f, num))
        except (ValueError, IndexError):
            continue  # Skip malformed filenames

    valid_files.sort(key=lambda item: item[1], reverse=True)

    # Extract filenames
    top_files = [f[0] for f in valid_files[:x]]

    # Optional: Warn if fewer than x valid files are found
    if len(top_files) < x:
        logging.warning('get_top_best_models: only %d valid files found, requested %d', len(top_files), x)
    return top_files

def get_model_by_modulo(file_list, prototype, modulo=75):
    """Select checkpoints at regular epoch intervals plus the final checkpoint.

    Always includes any checkpoint whose filename contains ``'final'``.  For
    the remaining files, selects epochs that are multiples of *modulo* and at
    most 550, plus the very last numbered epoch.

    Parameters
    ----------
    file_list : list of str
        Candidate checkpoint file paths.
    prototype : str
        Unused glob prototype (kept for API compatibility with
        :func:`find_best_performing_models`).
    modulo : int, optional
        Epoch interval. Default is 75.

    Returns
    -------
    list of str
        Selected checkpoint file paths.
    """
    selected = []
    my_dict = {}
    
    for file in file_list:
        if 'final' in os.path.basename(file):
            selected.append(file)
        else:
            basename = os.path.basename(file).split('.')[0]
            parts = basename.split('_')
            for part in parts:
                try:
                    number = int(part)
                    my_dict[number] = file
                    break 
                except Exception:
                    continue

    for index, (epoch, file) in enumerate(sorted(my_dict.items())):
        if epoch % modulo == 0 and epoch <= 550:
            selected.append(file)
        elif index == len(my_dict) - 1:
            selected.append(file)
    return selected
    
if __name__ == '__main__':

    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    index = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].lstrip('-').isdigit() else 0
    concurrent_workers = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].lstrip('-').isdigit() else 1
    overrides = parse_config_overrides(start_index=3)  # sys.argv[1]=index, sys.argv[2]=concurrent_workers, flags start at 3
    cfg = load_config(**overrides)
    eval_cfg = cfg['metrics']

    main(eval_cfg['models_dir'],
         eval_cfg['data_kwargs'], eval_cfg['model_kwargs'], eval_cfg['kwargs_source'],
         eval_cfg['total_workers'], eval_cfg['max_workers'], eval_cfg['frac'],
         condition, get_model_by_modulo, eval_cfg['n'], eval_cfg['model_prototype'],
         all_metrics_csv=eval_cfg['all_metrics_csv'],
         aggregated_metrics_csv=eval_cfg['aggregated_metrics_csv'],
         org_catalog_csv=eval_cfg['org_catalog_csv'],
         noisy_catalog_csv=eval_cfg['noisy_catalog_csv'],
         rec_catalog_csv=eval_cfg['rec_catalog_csv'],
         parallel=eval_cfg['parallel'], parallel_epoch=eval_cfg['parallel_epoch'], 
         scaling=eval_cfg['scaling'], index=index, concurrent_workers=
         concurrent_workers)