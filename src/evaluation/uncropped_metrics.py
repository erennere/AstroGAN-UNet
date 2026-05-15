"""Uncropped evaluation workflow for full-image source and flux metrics."""

import os, logging, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from src.evaluation.metrics import _reconstruct_patch, compare_images, find_best_performing_models, get_model_by_modulo, wrap_extract_sources, decide_scale, get_top_best_models
from src.data.create_dataset import crop_image_generator
from src.training.utils import open_fits, candidates_based_on_range, ensure_directory_exists, ensure_parent_dir_exists, build_checkpoint_custom_objects, load_checkpoint_model

from starter import load_config, parse_config_overrides  #sym:parse_config_overrides

def process_data(metadata_filepath, kwargs_data, kwargs_eval, output_filepath):
    """Build an evaluation metadata CSV using sigma-range candidate expansion.

    Filters the metadata CSV by PI last name, then delegates per-row candidate
    generation to :func:`~src.training.utils.candidates_based_on_range`.  The
    ``kwargs_data`` dict (config ``data`` section) is forwarded unchanged so
    all column-name and hyperparameter lookups are config-driven.

    Parameters
    ----------
    metadata_filepath : str
        Path to the metadata CSV file.
    kwargs_data : dict
        Config data section dict forwarded to
        :func:`~src.training.utils.candidates_based_on_range`.  Must contain
        at minimum ``'low'``, ``'name_col'``, ``'location_col'``,
        ``'dataset'``, ``'exposure_col'``, ``'stats_column_map'``, and
        ``'original_stats_prefix'``.
    kwargs_eval : dict
        Config evaluation section dict. Must contain
        ``'uncropped_last_name'``.
    output_filepath : str
        Path for the output CSV.

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame with one row per (image, sigma-candidate) pair.
    """
    low = kwargs_data['low']
    df = pd.read_csv(metadata_filepath)

    if kwargs_eval['filter_by_last_name']:
        last_names = kwargs_eval['last_name_filter_value']
        selected_df = df[df[kwargs_eval['last_name_col']].isin(last_names)]
    else:
        selected_df = df

    data = []
    for _, row in selected_df.iterrows():
        data.extend(candidates_based_on_range(row, kwargs_data))

    data_df = pd.DataFrame(data)
    filtered_data = data_df[data_df['new_exp_time'] >= low]
    ensure_parent_dir_exists(output_filepath)
    filtered_data.to_csv(output_filepath, index=False)
    return filtered_data

def process_subdf(sub_df, model_filepath, output_dir, kwargs, bins, save_eval_images):
    """Evaluate a model on a subset of the test images and return metrics/histograms.

    Parameters
    ----------
    sub_df : pandas.DataFrame
        Subset of the evaluation metadata DataFrame.
    model_filepath : str
        Path to the Keras ``.keras`` checkpoint file.
    output_dir : str
        Directory where per-image comparison plots are saved.
    kwargs : dict
        Source-detection and metrics parameters forwarded to
        :func:`~src.evaluation.metrics.compare_images`.
    bins : numpy.ndarray
        Histogram bin edges (log-spaced) for pixel-value histograms.

    Returns
    -------
    results_df : list of dict
        Per-crop metric dictionaries.
    all_hists : dict
        Mapping from ``exp_ratio`` to ``[original, noisy, reconstructed]`` histogram arrays.
    dfs : tuple
        ``(org_dfs, noisy_dfs, rec_dfs)`` concatenated source-catalogue DataFrames.
    """
    all_hists = {exp_ratio: [np.zeros(len(bins)-1), np.zeros(len(bins)-1), np.zeros(len(bins)-1)]
                 for exp_ratio in sub_df['exp_ratio'].unique()}
    results_df = []
    try:
        model = load_checkpoint_model(
            model_filepath,
            compile=False,
            custom_objects=build_checkpoint_custom_objects(),
        )
    except Exception as err:
        logging.error('process_subdf: failed to load model %s: %s', model_filepath, err)
        return None

    scales = decide_scale(model_filepath)  # None or (scaler, descaler)

    patch_size = tuple(kwargs['uncropped_patch_size'])
    stride = tuple(kwargs['uncropped_stride'])
    weighting = kwargs['uncropped_weighting']
    batch_size_inf = kwargs['uncropped_batch_size']
    nan_value = kwargs['nan_value']
    posinf_value = kwargs['posinf_value']
    neginf_value = kwargs['neginf_value']

    org_dfs = []
    rec_dfs = []
    noisy_dfs = []
    org_set = set()
    for _, row in sub_df.iterrows():
        location    = row['location']
        noisy_sigma = row[kwargs['sigma_key']]
        exp_time    = row['exp_time']
        new_exp_time = row['new_exp_time']
        image_id   = row['name']
        exp_ratio  = row['exp_ratio']
        image = open_fits(location, type_of_image=kwargs['type_of_image'])
        if image is None:
            continue

        i = 0

        for cropped_image in crop_image_generator(image, ps=patch_size[0]):
            cropped_image = np.nan_to_num(cropped_image, nan=nan_value, posinf=posinf_value, neginf=neginf_value)
            noisy_image = kwargs['noise_fn'](cropped_image, row, noisy_sigma)
            noisy_image = np.nan_to_num(noisy_image, nan=nan_value, posinf=posinf_value, neginf=neginf_value)

            rec_image = _reconstruct_patch(
                noisy_image, scales, model,
                use_mosaic=kwargs['uncropped_use_mosaic'], patch_size=patch_size, stride=stride,
                weighting=weighting, batch_size=batch_size_inf,
            )
            if rec_image is None:
                i += 1
                continue

            rec_image = np.nan_to_num(rec_image, nan=nan_value, posinf=posinf_value, neginf=neginf_value)
            noisy_image = np.nan_to_num(noisy_image, nan=nan_value, posinf=posinf_value, neginf=neginf_value)

            all_hists[exp_ratio][0] += np.histogram(cropped_image.flatten(), bins=bins)[0]
            all_hists[exp_ratio][1] += np.histogram(noisy_image.flatten(), bins=bins)[0]
            all_hists[exp_ratio][2] += np.histogram(rec_image.flatten(), bins=bins)[0]
            result = compare_images(cropped_image, noisy_image, rec_image, f'{image_id}_{str(i)}', exp_time, new_exp_time, output_dir, kwargs, save_eval_images)
            if result is not None:
                stats, (flux_rec, flux_org), (flux_error_rec, flux_error_org), (org_df, noisy_df, rec_df) = result 
                stats['flux_rec'] = flux_rec
                stats['flux_org'] = flux_org
                stats['flux_error_rec'] = flux_error_rec
                stats['flux_error_org'] = flux_error_org
                results_df.append(stats)

                if not org_df.empty:
                    if f'{image_id}_{str(i)}' not in org_set:
                        org_dfs.append(org_df)
                        org_set.add(f'{image_id}_{str(i)}')
                    noisy_dfs.append(noisy_df)
                    rec_dfs.append(rec_df)
            i += 1
    if len(org_dfs):
        org_dfs = pd.concat(org_dfs, ignore_index=False)
        noisy_dfs = pd.concat(noisy_dfs, ignore_index=False)
        rec_dfs = pd.concat(rec_dfs, ignore_index=False)
    else:
        org_dfs = pd.DataFrame(org_dfs)
        noisy_dfs = pd.DataFrame(noisy_dfs)
        rec_dfs = pd.DataFrame(rec_dfs)
    return results_df, all_hists, (org_dfs, noisy_dfs, rec_dfs)

def log_range(min_exp, max_exp):
    """Build a sorted array of decade-spaced values between exponent bounds."""

    data = []
    for i in range(min_exp, max_exp+1):
        for j in range(1,10):
            data.append(j*10.**i)
    data.sort()
    return np.array(data)

def main(N, model_filepath, metadata_filepath, output_dir, kwargs, workers, min_exp, max_exp, output_paths, save_eval_images):
    """Evaluate a model on a random sample of images and save results/histograms.

    Parameters
    ----------
    N : int
        Maximum number of images to evaluate.
    model_filepath : str
        Path to the Keras ``.keras`` checkpoint.
    metadata_filepath : str
        Path to the evaluation metadata CSV (must contain an ``'exp_ratio'`` column).
    output_dir : str
        Directory where CSVs, histograms, and comparison plots are saved.
    kwargs : dict
        Source-detection and metrics parameters forwarded to :func:`process_subdf`.
    workers : int, optional
        Number of parallel worker processes. Default is 4.
    min_exp : int, optional
        Minimum power-of-ten exponent for histogram bins. Default is ``-5``.
    max_exp : int, optional
        Maximum power-of-ten exponent for histogram bins. Default is ``5``.
    output_paths : dict
        Output filenames and templates. When provided, must contain:
        ``results_csv``, ``org_catalog_csv``, ``noisy_catalog_csv``,
        ``rec_catalog_csv``, ``hist_data_csv``, and ``hist_png_template``.
    """
    gal_df = pd.read_csv(metadata_filepath)
    gal_df = gal_df.sample(n=min(len(gal_df), N))

    ensure_directory_exists(output_dir)

    chunk_size = max(1, len(gal_df) // workers)
    sub_dfs = [gal_df.iloc[i:i + chunk_size] for i in range(0, len(gal_df), chunk_size)]
    bins = log_range(min_exp, max_exp)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_subdf, sub_df, model_filepath, output_dir, kwargs, bins, save_eval_images) for sub_df in sub_dfs]

    results = []
    all_hists = []
    org_dfs = []
    noisy_dfs = []
    rec_dfs = []
    org_set = set()
    for future in as_completed(futures):
        try:
            result = future.result()
            if result is not None:
                data, hists, (org_df, noise_df, rec_df) = result
                results.extend(data)
                all_hists.append(hists)

                if not org_df.empty:
                    temp = org_df[~org_df['image_id'].isin(org_set)]
                    org_set.update(temp['image_id'].tolist())
                    org_dfs.append(temp)
                    noisy_dfs.append(noise_df)
                    rec_dfs.append(rec_df)
        except Exception as err:
            print(err)

    results = pd.DataFrame(results)
    ensure_parent_dir_exists(output_paths['results_csv'])
    results.to_csv(output_paths['results_csv'], index=False)

    if len(org_dfs):
        org_dfs = pd.concat(org_dfs, ignore_index=False)
        noisy_dfs = pd.concat(noisy_dfs, ignore_index=False)
        rec_dfs = pd.concat(rec_dfs, ignore_index=False)
        ensure_parent_dir_exists(output_paths['org_catalog_csv'])
        ensure_parent_dir_exists(output_paths['noisy_catalog_csv'])
        ensure_parent_dir_exists(output_paths['rec_catalog_csv'])
        org_dfs.to_csv(output_paths['org_catalog_csv'], index=False)
        noisy_dfs.to_csv(output_paths['noisy_catalog_csv'], index=False)
        rec_dfs.to_csv(output_paths['rec_catalog_csv'], index=False)
    
    final_hists = {exp_ratio: [np.zeros(len(bins) - 1), np.zeros(len(bins) - 1), np.zeros(len(bins) - 1)]
                for exp_ratio in gal_df['exp_ratio'].unique()}
    for hist_group in all_hists:
        for exp_ratio, hists in hist_group.items():
            final_hists[exp_ratio][0] += hists[0]  # Original
            final_hists[exp_ratio][1] += hists[1]  # Noisy
            final_hists[exp_ratio][2] += hists[2]  # Reconstructed

    hist_data = []
    bin_edges = bins[1:] 
    colors = {
        'original': '#1f77b4',  # A softer blue (from seaborn palette)
        'noisy': '#ff7f0e',  # A warm orange
        'reconstructed': '#2ca02c'  # A fresh green
    }
    for exp_ratio, hists in final_hists.items():
        plt.figure(figsize=(8, 5)) 
        for hist, label in zip(hists, ['original', 'noisy', 'reconstructed']):
            plt.bar(
            bins[:-1], 
            hist, 
            width=np.diff(bins),  
            align='edge', 
            alpha=0.5, 
            label=label, 
            color=colors[label]
            )

            lower_bound = bins[0]
            for val, upper_bound in zip(hist, bin_edges):
                hist_data.append({
                    'label': label,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'exp_ratio': exp_ratio,
                    'count': val
                })
                lower_bound = upper_bound 

        plt.xlabel("Magnitude")
        plt.ylabel("Count")
        plt.xscale("log") 
        plt.title(f"Histogram for Exposure Ratio {int(exp_ratio)}")
        plt.legend()
        hist_png = output_paths['hist_png_template'].format(exp_ratio=int(exp_ratio), output_dir=output_dir)
        ensure_parent_dir_exists(hist_png)
        plt.savefig(hist_png, dpi=300)
        plt.close()

    hist_data = pd.DataFrame(hist_data)
    ensure_parent_dir_exists(output_paths['hist_data_csv'])
    hist_data.to_csv(output_paths['hist_data_csv'], index=False)

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    overrides = parse_config_overrides()
    cfg = load_config(**overrides)
    eval_cfg = cfg['evaluation']
    data_cfg = dict(eval_cfg['data_kwargs']['kwargs_data'])

    metadata_filepath = eval_cfg['metadata_filepath']
    uncropped_output_dir = eval_cfg['uncropped_output_dir']
    uncropped_combined_images_dir = eval_cfg['uncropped_combined_images_dir']
    output_paths = {
        'results_csv': str(Path(uncropped_output_dir) / eval_cfg['uncropped_results_csv']),
        'org_catalog_csv': str(Path(uncropped_output_dir) / eval_cfg['uncropped_org_catalog_csv']),
        'noisy_catalog_csv': str(Path(uncropped_output_dir) / eval_cfg['uncropped_noisy_catalog_csv']),
        'rec_catalog_csv': str(Path(uncropped_output_dir) / eval_cfg['uncropped_rec_catalog_csv']),
        'hist_data_csv': str(Path(uncropped_output_dir) / eval_cfg['uncropped_hist_data_csv']),
        'hist_png_template': str(Path(uncropped_output_dir) / eval_cfg['uncropped_hist_png_template']),
    }

    N               = eval_cfg['uncropped_n']
    output_filepath = eval_cfg['uncropped_sampled_data_csv']
    model_filepath = find_best_performing_models(
        eval_cfg['models_dir'],
        condition = lambda df: np.zeros(len(df)) == 0,
        filter_model=get_model_by_modulo,
        model_prototype=eval_cfg['model_prototype'],
        n=1,
        index=0,
        concurrent_workers=1
        )
    
    _ = process_data(
        metadata_filepath=metadata_filepath,
        kwargs_data=data_cfg,
        kwargs_eval=eval_cfg,
        output_filepath=output_filepath,
    )

    main(
        N, model_filepath, output_filepath, uncropped_combined_images_dir, eval_cfg['kwargs_source'],
        workers=eval_cfg['uncropped_workers'],
        min_exp=eval_cfg['hist_min_exp'],
        max_exp=eval_cfg['hist_max_exp'],
        output_paths=output_paths,
        save_eval_images=eval_cfg['uncropped_save_images']
    )
