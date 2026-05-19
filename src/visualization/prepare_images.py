"""Prepare composite image panels and source-overlays for qualitative review."""

import os, logging
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree
from src.evaluation.metrics import scale_image, wrap_extract_sources, get_test_images
from src.data.create_dataset import crop_image_generator
from src.training.utils import open_fits, ensure_parent_dir_exists, candidates_based_on_ratio, build_checkpoint_custom_objects, load_checkpoint_model, read_checkpoint_info, set_checkpoint_info_filename
from src.training.math_helpers import (
    create_simulated_image_gaussian,
    set_log_domain_clip_max,
)
from starter import parse_config_overrides, load_config
from src.evaluation.metrics import find_best_performing_models, get_model_by_modulo

def build_composite_axes(num_panels):
    """Build a matplotlib figure with one full-width original-image axes on top
    and a 2-row-per-panel grid below for (noisy, reconstructed) pairs.

    Parameters
    ----------
    num_panels : int
        Number of (noisy, reconstructed) column panels to allocate.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax0 : matplotlib.axes.Axes
        Top-row axes spanning all columns (original image).
    panel_axes : list of (Axes, Axes)
        One *(noisy_ax, rec_ax)* pair per panel.
    """
    columns = 4
    panel_rows = 0 if num_panels == 0 else int(np.ceil(num_panels / columns)) * 2
    total_rows = 1 + panel_rows

    fig = plt.figure(figsize=(16, max(6, 4 * total_rows)))
    gs = gridspec.GridSpec(total_rows, columns, height_ratios=[1] * total_rows)
    ax0 = plt.subplot(gs[0, :])

    panel_axes = []
    for index in range(num_panels):
        base_row = 1 + 2 * (index // columns)
        column = index % columns
        panel_axes.append((plt.subplot(gs[base_row, column]), plt.subplot(gs[base_row + 1, column])))

    return fig, ax0, panel_axes

def create_image(row, model, kwargs_data, ps=256):
    """
    Generator that yields (org, noisy_list, rec_list, gamma_list) tuples for
    every 256-px crop of the FITS image at row['location'].  Noise candidates
    are derived via candidates_based_on_ratio using the ratio hyperparameters
    embedded in kwargs_data (the data config section).
    """
    nan_value    = kwargs_data['nan_value']
    posinf_value = kwargs_data['posinf_value']
    neginf_value = kwargs_data['neginf_value']

    location = row['location']
    image = open_fits(location, type_of_image=kwargs_data['type_of_image'])
    if image is None:
        logging.warning(f"Could not open FITS file at {location}. Skipping this image.")
        return

    candidates = candidates_based_on_ratio(row, kwargs_data)
    if not len(candidates):
        logging.debug(f"No noise candidates for {location}. Skipping.")
        return
    candidates_df = pd.DataFrame(candidates)

    for _, cropped_image in enumerate(crop_image_generator(image, ps=ps)):
        noisy = []
        recs  = []
        gammas = []
        org = None

        cropped_image = np.nan_to_num(
            cropped_image, nan=nan_value, posinf=posinf_value, neginf=neginf_value
        )

        for _, candidate_row in candidates_df.iterrows():
            new_sigma  = candidate_row['combined_sigma']
            median_bkg = candidate_row['crop_median_bkg']
            noisy_image = create_simulated_image_gaussian(cropped_image, median_bkg, new_sigma)
            input_image = np.array([np.expand_dims(noisy_image, axis=-1)])
            rec_image   = model.predict(input_image, verbose=0)
            ratio = candidate_row['exp_ratio']
            if rec_image is not None:
                rec_image = rec_image[0, :, :, 0]
                gammas.append(ratio)
                noisy.append(noisy_image)
                recs.append(rec_image)
                org = cropped_image
        yield org, noisy, recs, gammas

def create_composite_plot(org, noisy, recs, gammas, label, output_filepath, output_dpi=None):
    """Save a composite PNG showing the original crop alongside noisy/reconstructed pairs.

    Parameters
    ----------
    org : np.ndarray
        Original 2-D image crop.
    noisy : list of np.ndarray
        Noise-degraded versions, one per gamma candidate.
    recs : list of np.ndarray
        Model reconstructions corresponding to each noisy image.
    gammas : list of float
        Exposure ratios (gamma) for each (noisy, rec) pair.
    label : str
        Figure title.
    output_filepath : str
        Destination PNG path; parent directories are created automatically.
    """
    ensure_parent_dir_exists(output_filepath)

    fig, ax0, panel_axes = build_composite_axes(len(gammas))
    org_scaled, vmin, vmax = scale_image(org)
    ax0.imshow(org_scaled, cmap='gray')
    ax0.set_yticks([])
    ax0.set_xticks([])

    for (noisy_ax, rec_ax), noisy_image, rec_image, gamma in zip(panel_axes, noisy, recs, gammas):
        noisy_scaled, _, _ = scale_image(noisy_image, vmin=vmin, vmax=vmax)
        noisy_ax.imshow(noisy_scaled, cmap='gray')
        noisy_ax.set_title(rf"Noisy Image, $\gamma$={int(gamma)}")
        noisy_ax.set_xticks([])
        noisy_ax.set_yticks([])

        rec_scaled, _, _ = scale_image(rec_image, vmin=vmin, vmax=vmax)
        rec_ax.imshow(rec_scaled, cmap='gray')
        rec_ax.set_title(rf"Reconstructed Image, $\gamma$={int(gamma)}")
        rec_ax.set_xticks([])
        rec_ax.set_yticks([])

    ax0.set_title(f"{label}", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_filepath, dpi=output_dpi)
    plt.close(fig)

def plot_source_comparison_sep(original_image, noisy_image, reconstructed_image,
                               x_org, y_org, x_rec, y_rec, x_noisy, y_noisy,
                               matched_indices_org, unmatched_indices_org,
                               matched_indices_rec, unmatched_indices_rec,
                               matched_indices_noisy, unmatched_indices_noisy,
                               org_a, org_b, rec_a, rec_b, noisy_a, noisy_b,
                               org_theta, rec_theta, noisy_theta):
    """Build ellipse lists for matched/unmatched source overlays across three images.

    Parameters
    ----------
    original_image, noisy_image, reconstructed_image : np.ndarray
        2-D image arrays (used only for validation checks).
    x_org, y_org : np.ndarray
        Source pixel coordinates detected on the original image.
    x_rec, y_rec : np.ndarray
        Source pixel coordinates detected on the reconstructed image.
    x_noisy, y_noisy : np.ndarray
        Source pixel coordinates detected on the noisy image.
    matched_indices_org, unmatched_indices_org : np.ndarray
        Integer index arrays partitioning original detections into matched/unmatched.
    matched_indices_rec, unmatched_indices_rec : np.ndarray
        Same for reconstructed detections.
    matched_indices_noisy, unmatched_indices_noisy : np.ndarray
        Same for noisy detections.
    org_a, org_b, org_theta : np.ndarray
        Semi-major axis, semi-minor axis, orientation (radians) for original sources.
    rec_a, rec_b, rec_theta : np.ndarray
        Same for reconstructed sources.
    noisy_a, noisy_b, noisy_theta : np.ndarray
        Same for noisy sources.

    Returns
    -------
    list of [org_ellipses, rec_ellipses, noisy_ellipses] or None on error.
        Each element is a list of :class:`~matplotlib.patches.Ellipse` objects.
        Red = matched, blue = unmatched-original, green = unmatched-rec, yellow = unmatched-noisy.
    """
    try:
        for image in [original_image, noisy_image, reconstructed_image]:
            if not isinstance(image, np.ndarray):
                print(f"Expected a NumPy array, got {type(image)}.")
                return
            if image.ndim < 2:
                print(f"Expected at least a 2D image, got shape {image.shape}.")
                return
            if not np.issubdtype(image.dtype, np.number):
                print(f"Image array must contain numerical values, got dtype {image.dtype}.")
                return

        org_ellipses = []
        for x, y, a, b, theta in zip(x_org[unmatched_indices_org], y_org[unmatched_indices_org], 
                                      org_a[unmatched_indices_org], org_b[unmatched_indices_org], 
                                      org_theta[unmatched_indices_org]):
            e = Ellipse(xy=(x, y), width=6*a, height=6*b, angle=np.degrees(theta), 
                        edgecolor='blue', facecolor='none', linewidth=1.5)
            org_ellipses.append(e)

        for x, y, a, b, theta in zip(x_org[matched_indices_org], y_org[matched_indices_org], 
                                      org_a[matched_indices_org], org_b[matched_indices_org], 
                                      org_theta[matched_indices_org]):
            e = Ellipse(xy=(x, y), width=6*a, height=6*b, angle=np.degrees(theta), 
                        edgecolor='red', facecolor='none', linewidth=1.5)
            org_ellipses.append(e)

        rec_ellipses = []
        for x, y, a, b, theta in zip(x_rec[unmatched_indices_rec], y_rec[unmatched_indices_rec], 
                                      rec_a[unmatched_indices_rec], rec_b[unmatched_indices_rec], 
                                      rec_theta[unmatched_indices_rec]):
            e = Ellipse(xy=(x, y), width=6*a, height=6*b, angle=np.degrees(theta), 
                        edgecolor='green', facecolor='none', linewidth=1.5)
            rec_ellipses.append(e)

        for x, y, a, b, theta in zip(x_rec[matched_indices_rec], y_rec[matched_indices_rec], 
                                      rec_a[matched_indices_rec], rec_b[matched_indices_rec], 
                                      rec_theta[matched_indices_rec]):
            e = Ellipse(xy=(x, y), width=6*a, height=6*b, angle=np.degrees(theta), 
                        edgecolor='red', facecolor='none', linewidth=1.5)
            rec_ellipses.append(e)

        noisy_ellipses = []
        for x, y, a, b, theta in zip(x_noisy[unmatched_indices_noisy], y_noisy[unmatched_indices_noisy],
                                     noisy_a[unmatched_indices_noisy], noisy_b[unmatched_indices_noisy],
                                     noisy_theta[unmatched_indices_noisy]):
            e = Ellipse(xy=(x, y), width=6*a, height=6*b, angle=np.degrees(theta),
                        edgecolor='yellow', facecolor='none', linewidth=1.5)
            noisy_ellipses.append(e)

        for x, y, a, b, theta in zip(x_noisy[matched_indices_noisy], y_noisy[matched_indices_noisy],
                                     noisy_a[matched_indices_noisy], noisy_b[matched_indices_noisy],
                                     noisy_theta[matched_indices_noisy]):
            e = Ellipse(xy=(x, y), width=6*a, height=6*b, angle=np.degrees(theta),
                        edgecolor='red', facecolor='none', linewidth=1.5)
            noisy_ellipses.append(e)

        return [org_ellipses, rec_ellipses, noisy_ellipses]

    except Exception as err:
        logging.warning(f"An error occurred while creating combined images: {err}")
        return

def compare_images(image_org, noisy_image, image_reconstructed, kwargs):
    """Run source detection on all three images and cross-match detections.

    Calls ``kwargs['func']`` (the SEP extraction callable) on each image, then
    uses a :class:`~scipy.spatial.cKDTree` nearest-neighbour search within
    ``kwargs['distance_threshold']`` pixels to classify each detection as
    matched or unmatched against the original catalogue.

    Parameters
    ----------
    image_org, noisy_image, image_reconstructed : np.ndarray
        2-D float arrays for the original, noisy, and reconstructed crops.
    kwargs : dict
        Must contain at minimum:
        - ``'func'`` — callable with signature ``(image, flag, kwargs)``
          (e.g. :func:`~src.evaluation.metrics.wrap_extract_sources`).
        - ``'distance_threshold'`` — pixel radius for cross-matching.

    Returns
    -------
    list or None
        Result of :func:`plot_source_comparison_sep` if successful, else ``None``.
    """
    if not isinstance(image_org, np.ndarray) or not isinstance(noisy_image, np.ndarray) or not isinstance(image_reconstructed, np.ndarray):
        return
    if image_org.shape != noisy_image.shape or noisy_image.shape != image_reconstructed.shape or image_org.shape != image_reconstructed.shape:
        return

    try:
        x_org, y_org, flux_org, flux_error_org, org_mask, org_a, org_b, org_theta, org_df = kwargs['func'](image_org, 'original', kwargs)
        x_rec, y_rec, flux_rec, flux_error_rec, rec_mask, rec_a, rec_b, rec_theta, rec_df = kwargs['func'](image_reconstructed, 'reconstructed', kwargs)
        x_noisy, y_noisy, flux_noisy, flux_error_noisy, noisy_mask, noisy_a, noisy_b, noisy_theta, noisy_df = kwargs['func'](noisy_image, 'noisy', kwargs)
    except Exception as err:
        logging.warning(f'could not find light sources: {err}')
        return

    valid_rec = ~(np.isnan(x_rec) | np.isnan(y_rec))
    valid_org = ~(np.isnan(x_org) | np.isnan(y_org))
    valid_noisy = ~(np.isnan(x_noisy) | np.isnan(y_noisy))
    valid_org_indices = np.flatnonzero(valid_org)
    valid_rec_indices = np.flatnonzero(valid_rec)
    valid_noisy_indices = np.flatnonzero(valid_noisy)

    matched_indices_image_org = np.array([], dtype=int)
    matched_indices_image_rec = np.array([], dtype=int)
    matched_indices_image_org_noisy = np.array([], dtype=int)
    matched_indices_image_noisy = np.array([], dtype=int)
    try:
        if len(valid_org_indices) and len(valid_rec_indices):
            tree_rec = cKDTree(np.column_stack((x_rec[valid_rec], y_rec[valid_rec])))
            distances, matched_rec_clean_indices = tree_rec.query(
                np.column_stack((x_org[valid_org], y_org[valid_org])),
                distance_upper_bound=kwargs['distance_threshold']
            )
            matched_org_clean_indices = np.where(distances < kwargs['distance_threshold'])[0]
            matched_indices_image_org = valid_org_indices[matched_org_clean_indices]
            matched_indices_image_rec = valid_rec_indices[matched_rec_clean_indices[matched_org_clean_indices]]

        if len(valid_org_indices) and len(valid_noisy_indices):
            tree_noisy = cKDTree(np.column_stack((x_noisy[valid_noisy], y_noisy[valid_noisy])))
            noisy_distances, matched_noisy_clean_indices = tree_noisy.query(
                np.column_stack((x_org[valid_org], y_org[valid_org])),
                distance_upper_bound=kwargs['distance_threshold']
            )
            matched_org_noisy_clean_indices = np.where(noisy_distances < kwargs['distance_threshold'])[0]
            matched_indices_image_org_noisy = valid_org_indices[matched_org_noisy_clean_indices]
            matched_indices_image_noisy = valid_noisy_indices[matched_noisy_clean_indices[matched_org_noisy_clean_indices]]
    except Exception as err:
        logging.warning(f"an error occurred while finding mutual sources: {err}")
        return

    unmatched_indices_image_org = np.setdiff1d(valid_org_indices, matched_indices_image_org)
    unmatched_indices_image_rec = np.setdiff1d(valid_rec_indices, matched_indices_image_rec)
    unmatched_indices_image_noisy = np.setdiff1d(valid_noisy_indices, matched_indices_image_noisy)

    return plot_source_comparison_sep(
        image_org,
        noisy_image,
        image_reconstructed,
        x_org,
        y_org,
        x_rec,
        y_rec,
        x_noisy,
        y_noisy,
        matched_indices_image_org,
        unmatched_indices_image_org,
        matched_indices_image_rec,
        unmatched_indices_image_rec,
        matched_indices_image_noisy,
        unmatched_indices_image_noisy,
        org_a,
        org_b,
        rec_a,
        rec_b,
        noisy_a,
        noisy_b,
        org_theta,
        rec_theta,
        noisy_theta,
    )

def create_composite_plot_detections(org, noisy, recs, gammas, ellipses, label, output_filepath, output_dpi=None):
    """Save a composite PNG overlaying detected-source ellipses on each image.

    Parameters
    ----------
    org : np.ndarray
        Original 2-D image crop (shown in the top row).
    noisy : list of np.ndarray
        Noise-degraded images, one per gamma candidate.
    recs : list of np.ndarray
        Model reconstructions corresponding to each noisy image.
    gammas : list of float
        Exposure ratios for each (noisy, rec) pair.
    ellipses : list
        One element per (noisy, rec) pair; each element is
        ``[org_ellipses, rec_ellipses, noisy_ellipses]`` as returned by
        :func:`plot_source_comparison_sep`.
    label : str
        Figure title.
    output_filepath : str
        Destination PNG path; parent directories are created automatically.
    """
    ensure_parent_dir_exists(output_filepath)

    fig, ax0, panel_axes = build_composite_axes(len(gammas))
    org, vmin, vmax = scale_image(org)
    ax0.imshow(org, cmap='gray')
    ax0.set_yticks([])
    ax0.set_xticks([])
    org_ellipses = ellipses[0][0]
    for ellipse in org_ellipses:
        ax0.add_patch(ellipse)

    for (noisy_ax, rec_ax), noisy_image, rec_image, gamma, ellipsex in zip(panel_axes, noisy, recs, gammas, ellipses):
        noisy_image, _, _ = scale_image(noisy_image, vmin=None, vmax=None)
        rec_image, _, _ = scale_image(rec_image, vmin=None, vmax=None)

        noisy_ax.imshow(noisy_image, cmap='gray')
        noisy_ax.set_title(rf"Noisy Image, $\gamma$={int(gamma)}")
        noisy_ax.set_xticks([])
        noisy_ax.set_yticks([])

        for ellipse in ellipsex[2]:
            noisy_ax.add_patch(ellipse)

        rec_ax.imshow(rec_image, cmap='gray')
        rec_ax.set_title(rf"Reconstructed Image, $\gamma$={int(gamma)}")
        rec_ax.set_xticks([])
        rec_ax.set_yticks([])

        for ellipse in ellipsex[1]:
            rec_ax.add_patch(ellipse)

    ax0.set_title(f"{label}", fontsize=16)
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='Matched Sources'),
        Line2D([0], [0], color='blue', lw=2, label='Unmatched Sources (Original)'),
        Line2D([0], [0], color='green', lw=2, label='Unmatched Sources (Reconstructed)'),
        Line2D([0], [0], color='yellow', lw=2, label='Unmatched Sources (Noisy)')
    ]

    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.01),
               ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(output_filepath, dpi=output_dpi)
    plt.close(fig)

def coordinate_detect_source(org, noisy, recs, gammas, label, kwargs, output_filepath, output_dpi=None):
    """Cross-match sources for every (noisy, rec) pair and save a detection overlay PNG.

    Iterates over (noisy, rec, gamma) triples, runs :func:`compare_images` on
    each, collects the resulting ellipse sets, and delegates to
    :func:`create_composite_plot_detections`.  Pairs for which source detection
    or cross-matching fails are silently skipped.

    Parameters
    ----------
    org : np.ndarray
        Original 2-D image crop.
    noisy : list of np.ndarray
        Noise-degraded images.
    recs : list of np.ndarray
        Reconstructed images.
    gammas : list of float
        Exposure ratios.
    label : str
        Figure title.
    kwargs : dict
        Forwarded verbatim to :func:`compare_images`; must include ``'func'``
        and ``'distance_threshold'``.
    output_filepath : str
        Destination PNG path.
    """
    all_ellipses = []
    filtered_noisy = []
    filtered_recs = []
    new_gammas = []
    for noisy_image, image_reconstructed, gamma in zip(noisy, recs, gammas):
        result = compare_images(org, noisy_image, image_reconstructed, kwargs)
        if result is not None:
            all_ellipses.append(result)
            filtered_noisy.append(noisy_image)
            filtered_recs.append(image_reconstructed)
            new_gammas.append(gamma)
    if len(all_ellipses):
        create_composite_plot_detections(
            org,
            filtered_noisy,
            filtered_recs,
            new_gammas,
            all_ellipses,
            label,
            output_filepath,
            output_dpi=output_dpi,
        )
    
def main():
    """Load config, sample evaluation images, and produce composite visualization PNGs.

    Reads ``visualization.prepare_images`` and ``data`` sections from config.yaml.

    Input files
    -----------
    ``<data_dir>/<metadata_filepath>`` : CSV
        Enriched metadata table; must contain ``location``, ``sci_actual_duration``,
        ``sci_targname``, and the dataset-name column.
    ``<model_dir>/<model_filepath>`` : Keras model checkpoint
        Trained denoising model loaded with ``compile=False``.

    Output files
    ------------
    ``<output_dir>/<label>_<crop_index>.png``
        Composite grid: original crop + (noisy, reconstructed) pairs per gamma.
    ``<output_dir>/<label>_<crop_index>_detections.png``
        Same grid with SEP source-detection ellipses overlaid.
    """
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    overrides = parse_config_overrides()
    cfg = load_config(**overrides)
    logging.info("Config loaded.")

    vis_cfg = cfg['prepare_images']
    data_cfg = dict(vis_cfg['data_kwargs'])
    set_checkpoint_info_filename(vis_cfg.get('checkpoint_info_filename', 'checkpoint_info.json'))
    set_log_domain_clip_max(data_cfg.get('log_domain_clip_max', 80.0))

    _output_dir     = vis_cfg['output_dir']

    _low = vis_cfg['low']
    _metadata_fp    = vis_cfg['metadata_filepath']
    _dataset_name   = vis_cfg['dataset']
    _sample_n       = vis_cfg['sample_n']
    _ps             = vis_cfg['ps']
    _kwargs_source  = vis_cfg['kwargs_source']
    _exp_col = vis_cfg['exp_column']
    _targ_col = vis_cfg['targ_col']

    plot_data_cfg = dict(data_cfg)
    plot_data_cfg['type_of_image'] = vis_cfg['type_of_image']
    for x in ('ratio_initial', 'ratio_count', 'ratio_growth'):
        plot_data_cfg[x] = vis_cfg[x]

    model_filepath = find_best_performing_models(
        vis_cfg['model_dir'],
        condition = lambda df: np.zeros(len(df)) == 0,
        filter_model=get_model_by_modulo,
        model_prototype=vis_cfg['model_prototype'],
        n=1,
        index=0,
        concurrent_workers=1
        )
    checkpoint_info = read_checkpoint_info(model_filepath)
    scaling = checkpoint_info.get('scaling', vis_cfg['scaling']) if isinstance(checkpoint_info, dict) else vis_cfg['scaling']

    if _metadata_fp is not None and not os.path.exists(_metadata_fp):
        logging.error(f"Metadata file {_metadata_fp} does not exist. Please check your config.")
        return
    elif _metadata_fp is not None:
        logging.info(f"Loading metadata from {_metadata_fp}.")
        metadata_df = pd.read_csv(_metadata_fp)
    else:
        logging.info("No metadata file provided; loading test images directly.")
        metadata_df = get_test_images(None, plot_data_cfg, scaling=scaling)
        metadata_df = metadata_df[metadata_df['location'].str.contains('test', na=False)].sample(frac=1)

    selected_metadata_df = metadata_df[
        metadata_df[_exp_col] / plot_data_cfg['ratio_initial']*(plot_data_cfg['ratio_initial']**plot_data_cfg['ratio_growth']) >= _low
    ].sample(n=_sample_n)

    logging.info(f"Sampled {len(selected_metadata_df)} images.")
    logging.info(f"Loading model from {model_filepath}.")
    model = load_checkpoint_model(
        model_filepath,
        compile=False,
        custom_objects=build_checkpoint_custom_objects(),
    )
    logging.info("Model loaded.")

    for (_, row), label in zip(
        selected_metadata_df.iterrows(),
        [f"ID{i+1}" for i in range(len(selected_metadata_df))]
    ):
        target = row[_targ_col]
        id_ = row[_dataset_name]
        combined_label = f'target: {target}, sci_data_set_name: {id_}'
        logging.info(f"Processing {label}: {combined_label}")
        for index_2, (org, noisy, recs, gammas) in enumerate(
            create_image(row, model, plot_data_cfg, ps=_ps)
        ):
            if org is not None:
                try:
                    create_composite_plot(
                        org, noisy, recs, gammas, combined_label,
                        os.path.join(_output_dir, f'{label}_{index_2}.png'),
                        vis_cfg.get('output_dpi', None),
                    )
                    coordinate_detect_source(
                        org, noisy, recs, gammas, combined_label, _kwargs_source,
                        os.path.join(_output_dir, f'{label}_{index_2}_detections.png'),
                        vis_cfg.get('output_dpi', None),
                    )
                    logging.info(f"  Saved crop {index_2}: {len(gammas)} gamma(s).")
                except Exception as err:
                    logging.warning(f"Visualization error for {label}_{index_2}: {err}")

if __name__ == "__main__":
    main()
