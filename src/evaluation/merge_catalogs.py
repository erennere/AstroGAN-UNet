"""Merge uncropped source catalogs into a single photometric parquet table."""

import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

from starter import load_config, parse_config_overrides
from src.training.utils import ensure_parent_dir_exists


def merge_based_on_proximity(df_rec, df_noise, df_org, threshold=3.0):
    """Merge reconstructed/noisy/original source catalogs by spatial proximity.

    For each ``image_id`` and ``new_exp_time`` group, reconstructed detections
    are matched to the nearest original and noisy detections within
    ``threshold`` pixels. Unmatched rows are retained with missing values for
    absent counterparts.

    Parameters
    ----------
    df_rec : pandas.DataFrame
        Reconstructed-source catalog. Must include ``image_id``,
        ``new_exp_time``, ``x``, ``y``.
    df_noise : pandas.DataFrame
        Noisy-source catalog with the same key columns as ``df_rec``.
    df_org : pandas.DataFrame
        Original-source catalog. Must include ``image_id``, ``x``, ``y``.
    threshold : float, optional
        Maximum nearest-neighbor distance in pixels for a valid match.

    Returns
    -------
    pandas.DataFrame
        Wide merged table with suffixes ``_rec``, ``_noise``, and ``_org``.
    """
    merged_results = []

    #for image_id in tqdm(df_rec['image_id'].unique(), desc="Processing image_id", unit="image_id"):
    for image_id in df_rec['image_id'].unique():
        df_rec_filtered = df_rec[df_rec['image_id'] == image_id]
        df_noise_filtered = df_noise[df_noise['image_id'] == image_id]
        df_org_filtered = df_org[df_org['image_id'] == image_id].reset_index(drop=True)

        exp_times = df_rec_filtered['new_exp_time'].unique()
        for new_exp_time in exp_times:
            df_rec_second_filtered = df_rec_filtered[df_rec_filtered['new_exp_time'] == new_exp_time].reset_index(drop=True)
            df_noise_second_filtered = df_noise_filtered[df_noise_filtered['new_exp_time'] == new_exp_time].reset_index(drop=True)

            rec_coords = df_rec_second_filtered[['x', 'y']].values
            noise_coords = df_noise_second_filtered[['x', 'y']].values
            org_coords = df_org_filtered[['x', 'y']].values

            merged_rows = []

            # Match rec with org
            if len(org_coords) > 0 and len(rec_coords) > 0:
                tree = KDTree(org_coords)
                dist, indices_org = tree.query(rec_coords, k=1)
                valid_matches_org = dist[:, 0] <= threshold
            else:
                indices_org = np.empty((len(rec_coords), 1), dtype=int)
                valid_matches_org = np.zeros(len(rec_coords), dtype=bool)

            for i, row_rec in df_rec_second_filtered.iterrows():
                row_rec_renamed = row_rec.rename(lambda x: x + '_rec' if x != 'image_id' else x).to_dict()

                if len(org_coords) > 0 and valid_matches_org[i]:
                    row_org = df_org_filtered.iloc[indices_org[i][0]]
                    row_org_renamed = row_org.rename(lambda x: x + '_org' if x != 'image_id' else x).to_dict()
                    merged_row = {**row_rec_renamed, **row_org_renamed}
                    merged_row['i'] = i
                    merged_rows.append(merged_row)
                else:
                    # If no match in org, fill org columns with NaN
                    for col in df_org.columns:
                        if col != 'image_id':
                            row_rec_renamed[col + '_org'] = np.nan
                    row_rec_renamed['i'] = i
                    merged_rows.append(row_rec_renamed)

            merged_rows_df = pd.DataFrame(merged_rows)

            # Match merged rows with noise (nearest-neighbor only)
            if len(noise_coords) > 0 and len(rec_coords) > 0:
                tree = KDTree(noise_coords)
                dist, indices_noise = tree.query(rec_coords, k=1)
                valid_matches_noise = dist[:, 0] <= threshold
            else:
                indices_noise = np.empty((len(rec_coords), 1), dtype=int)
                valid_matches_noise = np.zeros(len(rec_coords), dtype=bool)

            for _, row_rec in merged_rows_df.iterrows():
                i = int(row_rec['i'])
                row_rec_dict = row_rec.to_dict()
                if len(noise_coords) > 0 and valid_matches_noise[i]:
                    row_noise = df_noise_second_filtered.iloc[indices_noise[i][0]]
                    row_noise_renamed = row_noise.rename(lambda x: x + '_noise' if x != 'image_id' else x).to_dict()
                    merged_row = {**row_rec_dict, **row_noise_renamed}
                    merged_results.append(merged_row)
                else:
                    for col in df_noise.columns:
                        if col != 'image_id':
                            row_rec_dict[col + '_noise'] = np.nan
                    merged_results.append(row_rec_dict)

            matched_org_indices = set(indices_org[valid_matches_org, 0].tolist()) if len(org_coords) > 0 else set()
            matched_noise_indices = set(indices_noise[valid_matches_noise, 0].tolist()) if len(noise_coords) > 0 else set()

            unmatched_org_indices = set(range(len(df_org_filtered))) - matched_org_indices
            unmatched_noise_indices = set(range(len(df_noise_second_filtered))) - matched_noise_indices

            for index in unmatched_org_indices:
                row_org = df_org_filtered.iloc[index]
                new_row = row_org.rename(lambda x: x + '_org' if x != 'image_id' else x).to_dict()
                for col in df_noise.columns:
                    if col != 'image_id':
                        new_row[col + '_noise'] = np.nan
                for col in df_rec.columns:
                    if col != 'image_id':
                        new_row[col + '_rec'] = np.nan
                merged_results.append(new_row)

            for index in unmatched_noise_indices:
                row_noise = df_noise_second_filtered.iloc[index]
                new_row = row_noise.rename(lambda x: x + '_noise' if x != 'image_id' else x).to_dict()
                for col in df_org.columns:
                    if col != 'image_id':
                        new_row[col + '_org'] = np.nan
                for col in df_rec.columns:
                    if col != 'image_id':
                        new_row[col + '_rec'] = np.nan
                merged_results.append(new_row)

    return pd.DataFrame(merged_results)


def process(noise_csv, org_csv, rec_csv, workers, output_parquet, threshold=3.0):
    """Run multiprocessing catalog merge and write a parquet output.

    Parameters
    ----------
    noise_csv : str
        Path to noisy catalog CSV.
    org_csv : str
        Path to original catalog CSV.
    rec_csv : str
        Path to reconstructed catalog CSV.
    workers : int
        Number of worker processes.
    output_parquet : str
        Destination parquet path.
    threshold : float, optional
        Pixel-distance threshold passed to :func:`merge_based_on_proximity`.
    """
    noise_df = pd.read_csv(noise_csv)
    org_df = pd.read_csv(org_csv)
    rec_df = pd.read_csv(rec_csv)

    unique_image_ids = list(
        set(org_df['image_id'].unique()).union(
            set(rec_df['image_id'].unique()),
            set(noise_df['image_id'].unique()),
        )
    )
    workers = max(int(workers), 1)
    chunks = np.array_split(unique_image_ids, workers)
    all_data = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = []
        for chunk in chunks:
            ids = set(chunk)
            chunk_org_df = org_df[org_df['image_id'].isin(ids)].reset_index(drop=True)
            chunk_rec_df = rec_df[rec_df['image_id'].isin(ids)].reset_index(drop=True)
            chunk_noise_df = noise_df[noise_df['image_id'].isin(ids)].reset_index(drop=True)

            futures.append(
                executor.submit(
                    merge_based_on_proximity,
                    chunk_rec_df,
                    chunk_noise_df,
                    chunk_org_df,
                    threshold,
                )
            )

    for future in as_completed(futures):
        try:
            result = future.result()
            if result is not None:
                all_data.append(result)
        except Exception as err:
            print(f'Error processing chunk: {err}')

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        ensure_parent_dir_exists(output_parquet)
        final_df.to_parquet(output_parquet, index=False)


def _run_from_config(eval_cfg):
    """Build file paths from evaluation config and execute one merge job.

    Expects uncropped evaluation catalog filenames in
    ``uncropped_*_catalog_csv`` and output filename in
    ``photometrical_data_filename`` relative to
    ``uncropped_output_dir``.
    """
    file_dir = eval_cfg['uncropped_output_dir']
    rec_filename = eval_cfg['uncropped_rec_catalog_csv']
    noise_filename = eval_cfg['uncropped_noisy_catalog_csv']
    org_filename = eval_cfg['uncropped_org_catalog_csv']
    photometrical_data_filename = eval_cfg['photometrical_data_filename']

    rec_filepath = os.path.join(file_dir, rec_filename)
    noise_filepath = os.path.join(file_dir, noise_filename) 
    org_filepath = os.path.join(file_dir, org_filename)
    photometrical_data_filepath = os.path.join(file_dir, photometrical_data_filename)

    workers = int(eval_cfg.get('merge_catalog_workers', eval_cfg['workers']))
    threshold = float(eval_cfg['merge_catalog_threshold'])

    process(
            noise_csv=noise_filepath,
            org_csv=org_filepath,
            rec_csv=rec_filepath,
            workers=workers,
            threshold=threshold,
            output_parquet=photometrical_data_filepath,
        )

if __name__ == '__main__':
    """CLI entrypoint using the shared config loader and override parser."""
    overrides = parse_config_overrides()
    cfg = load_config(**overrides)
    _run_from_config(cfg['evaluation'])
