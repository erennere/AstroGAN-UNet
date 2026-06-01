"""Uncropped evaluation workflow for full-image source and flux metrics."""

import os, logging, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from src.evaluation.metrics import _reconstruct_patch, compare_images, condition, find_best_performing_models, get_model_by_modulo, wrap_extract_sources, decide_scale, parse_runtime_selector_cli_args
from src.data.create_dataset import crop_image_generator
from src.training.math_helpers import set_log_domain_clip_max
from src.training.utils import open_fits, candidates_based_on_range, ensure_directory_exists, ensure_parent_dir_exists, build_checkpoint_custom_objects, load_checkpoint_model, set_checkpoint_info_filename

from starter import load_config, parse_config_overrides, _decode_models_dir  #sym:parse_config_overrides

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
    if isinstance(kwargs, tuple) and len(kwargs) == 1 and isinstance(kwargs[0], dict):
        kwargs = kwargs[0]
    if not isinstance(kwargs, dict):
        raise TypeError(f'process_subdf expected kwargs as dict, got {type(kwargs).__name__}.')

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

    try:
        scales = decide_scale(model_filepath)  # None or (scaler, descaler)
    except Exception as err:
        logging.error('process_subdf: failed to resolve scaling for model %s: %s', model_filepath, err)
        return None

    try:
        patch_size = tuple(int(x) for x in kwargs['uncropped_patch_size'])
        stride = tuple(int(x) for x in kwargs['uncropped_stride'])
        weighting = kwargs['uncropped_weighting']
        batch_size_inf = int(kwargs['uncropped_batch_size'])
        nan_value = kwargs['nan_value']
        posinf_value = kwargs['posinf_value']
        neginf_value = kwargs['neginf_value']
    except Exception as err:
        logging.error('process_subdf: invalid uncropped runtime kwargs: %s', err)
        return None

    def _record_outputs(crop_payload, rec_image):
        crop_tag = crop_payload['crop_tag']
        location = crop_payload['location']
        exp_ratio = crop_payload['exp_ratio']
        cropped_image = crop_payload['cropped_image']
        noisy_image = crop_payload['noisy_image']
        exp_time = crop_payload['exp_time']
        new_exp_time = crop_payload['new_exp_time']

        if rec_image is None:
            logging.warning('process_subdf: reconstruction returned None for %s (%s)', crop_tag, location)
            return

        try:
            rec_image = np.nan_to_num(rec_image, nan=nan_value, posinf=posinf_value, neginf=neginf_value)
            noisy_image = np.nan_to_num(noisy_image, nan=nan_value, posinf=posinf_value, neginf=neginf_value)

            all_hists[exp_ratio][0] += np.histogram(cropped_image.flatten(), bins=bins)[0]
            all_hists[exp_ratio][1] += np.histogram(noisy_image.flatten(), bins=bins)[0]
            all_hists[exp_ratio][2] += np.histogram(rec_image.flatten(), bins=bins)[0]
        except Exception as err:
            logging.warning('process_subdf: histogram accumulation failed for %s (%s): %s', crop_tag, location, err)
            return

        try:
            result = compare_images(
                cropped_image,
                noisy_image,
                rec_image,
                crop_tag,
                exp_time,
                new_exp_time,
                output_dir,
                kwargs,
                save_eval_images,
            )
        except Exception as err:
            logging.warning('process_subdf: compare_images failed for %s (%s): %s', crop_tag, location, err)
            return

        if result is None:
            return

        try:
            stats, (flux_rec, flux_org), (flux_error_rec, flux_error_org), (org_df, noisy_df, rec_df) = result
            stats['flux_rec'] = flux_rec
            stats['flux_org'] = flux_org
            stats['flux_error_rec'] = flux_error_rec
            stats['flux_error_org'] = flux_error_org
            results_df.append(stats)

            if not org_df.empty:
                if crop_tag not in org_set:
                    org_dfs.append(org_df)
                    org_set.add(crop_tag)
                noisy_dfs.append(noisy_df)
                rec_dfs.append(rec_df)
        except Exception as err:
            logging.warning('process_subdf: failed to record outputs for %s (%s): %s', crop_tag, location, err)

    def _flush_non_mosaic_batch(batch_items):
        if not batch_items:
            return

        reconstructed_images = [None] * len(batch_items)
        try:
            if scales is not None:
                scale, descale = scales
                scaled_batch = []
                scaled_args = []
                valid_indices = []
                for idx, item in enumerate(batch_items):
                    try:
                        scale_result = scale(item['noisy_image'])
                        if scale_result is None:
                            logging.warning('process_subdf: scaling returned None for %s (%s)', item['crop_tag'], item['location'])
                            continue
                        scaled_batch.append(np.expand_dims(scale_result[0], axis=-1))
                        scaled_args.append(scale_result[1:])
                        valid_indices.append(idx)
                    except Exception as err:
                        logging.warning('process_subdf: scaling failed for %s (%s): %s', item['crop_tag'], item['location'], err)

                if scaled_batch:
                    raw_predictions = model.predict(np.asarray(scaled_batch), batch_size=batch_size_inf, verbose=0)
                    for pred_idx, item_idx in enumerate(valid_indices):
                        try:
                            reconstructed_images[item_idx] = descale(raw_predictions[pred_idx, :, :, 0], *scaled_args[pred_idx])
                        except Exception as err:
                            payload = batch_items[item_idx]
                            logging.warning('process_subdf: descale failed for %s (%s): %s', payload['crop_tag'], payload['location'], err)
            else:
                batch_input = np.asarray([np.expand_dims(item['noisy_image'], axis=-1) for item in batch_items])
                raw_predictions = model.predict(batch_input, batch_size=batch_size_inf, verbose=0)
                for idx in range(len(batch_items)):
                    reconstructed_images[idx] = raw_predictions[idx, :, :, 0]
        except Exception as err:
            logging.warning('process_subdf: batched non-mosaic reconstruction failed: %s', err)

        for payload, rec_image in zip(batch_items, reconstructed_images):
            _record_outputs(payload, rec_image)

    org_dfs = []
    rec_dfs = []
    noisy_dfs = []
    org_set = set()
    use_mosaic = bool(kwargs['uncropped_use_mosaic'])
    for _, row in sub_df.iterrows():
        try:
            location = row['location']
            noisy_sigma = row[kwargs['sigma_key']]
            exp_time = row['exp_time']
            new_exp_time = row['new_exp_time']
            image_id = row['name']
            exp_ratio = row['exp_ratio']
        except Exception as err:
            logging.warning('process_subdf: skipping row due to missing required fields: %s', err)
            continue

        try:
            image = open_fits(location, type_of_image=kwargs['type_of_image'])
        except Exception as err:
            logging.warning('process_subdf: failed to open FITS %s: %s', location, err)
            continue
        if image is None:
            logging.warning('process_subdf: open_fits returned None for %s', location)
            continue

        i = 0
        pending_non_mosaic = []
        try:
            for cropped_image in crop_image_generator(image, ps=patch_size[0]):
                crop_tag = f'{image_id}_{str(i)}'
                try:
                    cropped_image = np.nan_to_num(cropped_image, nan=nan_value, posinf=posinf_value, neginf=neginf_value)
                    noisy_image = kwargs['noise_fn'](cropped_image, row, noisy_sigma)
                    noisy_image = np.nan_to_num(noisy_image, nan=nan_value, posinf=posinf_value, neginf=neginf_value)

                    payload = {
                        'crop_tag': crop_tag,
                        'location': location,
                        'exp_ratio': exp_ratio,
                        'cropped_image': cropped_image,
                        'noisy_image': noisy_image,
                        'exp_time': exp_time,
                        'new_exp_time': new_exp_time,
                    }

                    if use_mosaic:
                        rec_image = _reconstruct_patch(
                            noisy_image,
                            scales,
                            model,
                            use_mosaic=use_mosaic,
                            patch_size=patch_size,
                            stride=stride,
                            weighting=weighting,
                            batch_size=batch_size_inf,
                            gaussian_sigma=kwargs['gaussian_sigma'],
                        )
                        _record_outputs(payload, rec_image)
                    else:
                        pending_non_mosaic.append(payload)
                        if len(pending_non_mosaic) >= batch_size_inf:
                            _flush_non_mosaic_batch(pending_non_mosaic)
                            pending_non_mosaic = []
                except Exception as err:
                    logging.warning('process_subdf: crop processing failed for %s (%s): %s', crop_tag, location, err)
                finally:
                    i += 1
        except Exception as err:
            logging.warning('process_subdf: crop iteration failed for image %s (%s): %s', image_id, location, err)

        if not use_mosaic and pending_non_mosaic:
            _flush_non_mosaic_batch(pending_non_mosaic)
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


def write_histogram_outputs(all_hists, exp_ratios, bins, output_paths, output_dir):
    """Aggregate histogram counts across workers and write PNG/CSV outputs."""
    final_hists = {
        exp_ratio: [np.zeros(len(bins) - 1), np.zeros(len(bins) - 1), np.zeros(len(bins) - 1)]
        for exp_ratio in exp_ratios
    }
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

        plt.xlabel('Magnitude')
        plt.ylabel('Count')
        plt.xscale('log')
        plt.title(f'Histogram for Exposure Ratio {int(exp_ratio)}')
        plt.legend()
        hist_png = output_paths['hist_png_template'].format(exp_ratio=int(exp_ratio), output_dir=output_dir)
        ensure_parent_dir_exists(hist_png)
        plt.savefig(hist_png, dpi=300)
        plt.close()

    hist_data = pd.DataFrame(hist_data)
    ensure_parent_dir_exists(output_paths['hist_data_csv'])
    hist_data.to_csv(output_paths['hist_data_csv'], index=False)

def orchestrate_single_run(N, model_filepath, metadata_filepath, output_dir, 
                           kwargs, workers, min_exp, max_exp, output_paths,
                          save_eval_images, single_parallel=False, 
                          write_histograms=True, save_combined_images=True,
                          overwrite=True):
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
    N = int(N)
    workers = int(workers)
    min_exp = int(min_exp)
    max_exp = int(max_exp)
    if workers <= 0:
        raise ValueError('workers must be a positive integer.')

    if not overwrite and os.path.exists(output_paths['results_csv']):
        logging.info('orchestrate_single_run: %s already exists and overwrite=False, skipping.', output_paths['results_csv'])
        return

    gal_df = pd.read_csv(metadata_filepath)
    gal_df = gal_df.sample(n=min(len(gal_df), N))

    if save_combined_images:
        ensure_directory_exists(output_dir)
    results = []
    all_hists = []
    org_dfs = []
    noisy_dfs = []
    rec_dfs = []
    org_set = set()

    chunk_size = max(1, len(gal_df) // workers)
    sub_dfs = [gal_df.iloc[i:i + chunk_size] for i in range(0, len(gal_df), chunk_size)]
    bins = log_range(min_exp, max_exp)

    if not single_parallel:
        for sub_df in sub_dfs:
            result = process_subdf(sub_df, model_filepath, output_dir, kwargs, bins, save_eval_images and save_combined_images)
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
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_subdf, sub_df, model_filepath, output_dir, kwargs, bins, save_eval_images and save_combined_images) for sub_df in sub_dfs]

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
                logging.warning('orchestrate_single_run: a worker process failed with error: %s', err)
    
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

    if write_histograms:
        write_histogram_outputs(
            all_hists=all_hists,
            exp_ratios=gal_df['exp_ratio'].unique(),
            bins=bins,
            output_paths=output_paths,
            output_dir=output_dir,
        )


def orchestrate_uncropped_evaluation(eval_cfg, data_cfg, metadata_filepath, model_filepath,
                                     data_alias_enriched_hex=None, model_alias_hex=None, epoch='*'
                                     ):
    """Prepare uncropped output paths, sample metadata, and run one evaluation pass."""
    data_alias_enriched_hex = str(data_alias_enriched_hex)
    model_alias_hex = str(model_alias_hex)
    epoch = str(epoch)
    uncropped_output_dir = eval_cfg['uncropped_output_dir'].replace('#', data_alias_enriched_hex).replace('&', model_alias_hex).replace('*', epoch)
    uncropped_combined_images_dir = eval_cfg['uncropped_combined_images_dir'].replace('#', data_alias_enriched_hex).replace('&', model_alias_hex).replace('*', epoch)
    output_filepath = eval_cfg['uncropped_sampled_data_csv'].replace('#', data_alias_enriched_hex).replace('&', model_alias_hex).replace('*', epoch)

    output_paths = {
        'results_csv': str(Path(uncropped_output_dir) / eval_cfg['uncropped_results_csv'].replace('&', model_alias_hex)),
        'org_catalog_csv': str(Path(uncropped_output_dir) / eval_cfg['uncropped_org_catalog_csv'].replace('&', model_alias_hex)),
        'noisy_catalog_csv': str(Path(uncropped_output_dir) / eval_cfg['uncropped_noisy_catalog_csv'].replace('&', model_alias_hex)),
        'rec_catalog_csv': str(Path(uncropped_output_dir) / eval_cfg['uncropped_rec_catalog_csv'].replace('&', model_alias_hex)),
        'hist_data_csv': str(Path(uncropped_output_dir) / eval_cfg['uncropped_hist_data_csv'].replace('&', model_alias_hex)),
        'hist_png_template': str(Path(uncropped_output_dir) / eval_cfg['uncropped_hist_png_template'].replace('&', model_alias_hex)),
    }

    n_samples = eval_cfg['uncropped_n']
    _ = process_data(
        metadata_filepath=metadata_filepath,
        kwargs_data=data_cfg,
        kwargs_eval=eval_cfg,
        output_filepath=output_filepath,
    )
    
    orchestrate_single_run(
        n_samples, model_filepath, output_filepath, uncropped_combined_images_dir, eval_cfg['kwargs_source'],
        workers=eval_cfg['uncropped_workers'],
        min_exp=eval_cfg['hist_min_exp'],
        max_exp=eval_cfg['hist_max_exp'],
        output_paths=output_paths,
        save_eval_images=eval_cfg['uncropped_save_images'],
        single_parallel=eval_cfg['uncropped_single_parallel'],
        write_histograms=eval_cfg['uncropped_write_histograms'],
        save_combined_images=eval_cfg['uncropped_save_combined_images'],
        overwrite=eval_cfg['uncropped_overwrite'],
    )

def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    selector_cli = parse_runtime_selector_cli_args()
    overrides = parse_config_overrides(start_index=selector_cli['cursor'])

    cfg = load_config(**overrides)
    eval_cfg = cfg['uncropped_metrics']
    data_cfg = dict(eval_cfg['data_kwargs']['kwargs_data'])
    set_checkpoint_info_filename(eval_cfg['checkpoint_info_filename'])
    set_log_domain_clip_max(data_cfg['log_domain_clip_max'])

    condition_kwargs = {
        'data_alias_enriched_hex': (
            selector_cli['data_alias_enriched_hex']
            if selector_cli['data_alias_enriched_hex'] is not None
            else data_cfg['data_alias_enriched_hex']
        ),
        'model_alias_hex': selector_cli['model_alias_hex'],
        'epoch': selector_cli['epoch'],
        'config_model_alias_hex': data_cfg['model_alias_hex'],
    }

    metadata_filepath = eval_cfg['metadata_filepath']
    models_dict = find_best_performing_models(
        eval_cfg['models_dir'],
        condition = condition,
        filter_model=get_model_by_modulo,
        model_prototype=eval_cfg['model_prototype'],
        modulo=eval_cfg['modulo'],
        index=selector_cli['index'],
        concurrent_workers=selector_cli['concurrent_workers'],
        condition_kwargs=condition_kwargs
    )

    jobs = []
    for model_dir in models_dict.keys():
        logging.warning('Processing model directory: %s', model_dir)
        model_df = models_dict[model_dir]
        for _, row in model_df.iterrows():
            decoded_model_dir = os.path.dirname(os.path.dirname(row.filepath))
            jobs.append((row.filepath, row.epoch, 
                         _decode_models_dir(decoded_model_dir)['model_alias_hex']))
        
    with ProcessPoolExecutor(max_workers=eval_cfg['max_workers']) as executor:
        futures = []
        for model_filepath, epoch, model_alias_hex in jobs:
            future = executor.submit(
                orchestrate_uncropped_evaluation,
                eval_cfg=eval_cfg,
                data_cfg=data_cfg,
                metadata_filepath=metadata_filepath,
                model_filepath=model_filepath,
                data_alias_enriched_hex=condition_kwargs['data_alias_enriched_hex'],
                model_alias_hex=model_alias_hex,
                epoch=epoch
            )
            futures.append(future)

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as err:
                logging.warning('A worker process failed with error: %s', err)

if __name__ == '__main__':
    main()
