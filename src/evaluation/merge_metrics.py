import os
import glob
import logging
import pandas as pd

from starter import load_config, parse_config_overrides, _decode_models_dir
from src.evaluation.metrics import parse_runtime_selector_cli_args


def _exclude_merged_outputs(filepaths):
    """Exclude previously merged outputs so reruns do not re-ingest them."""
    excluded_names = {'aggregated_metrics_merged.csv', 'metrics_merged.csv'}
    return [path for path in filepaths if os.path.basename(path) not in excluded_names]

def merge_metrics(filepaths, models_root_dir=None):
    results = []
    for filepath in filepaths:
        try:
            df = pd.read_csv(filepath)
            if not df.columns.is_unique:
                dup_cols = df.columns[df.columns.duplicated()].tolist()
                logging.warning(
                    'Duplicate columns in %s: %s. Keeping first occurrence.',
                    filepath,
                    dup_cols,
                )
                df = df.loc[:, ~df.columns.duplicated(keep='first')]
            if 'model' not in df.columns:
                logging.warning(f"'model' column not found in {filepath}. Skipping model decode.")
            else:
                def _decode(model_path):
                    if models_root_dir and not os.path.isabs(model_path):
                        model_path = os.path.join(models_root_dir, model_path)
                    return _decode_models_dir(model_path)
                model_dicts = df['model'].map(_decode)
                decoded_df = pd.DataFrame(model_dicts.tolist(), index=df.index)
                overlapping_cols = [col for col in decoded_df.columns if col in df.columns]
                if overlapping_cols:
                    logging.info(
                        'Overwriting existing columns in %s from decoded model metadata: %s',
                        filepath,
                        overlapping_cols,
                    )
                    df = df.drop(columns=overlapping_cols)
                df = pd.concat([df, decoded_df], axis=1)
            results.append(df)
        except Exception as e:
            logging.warning(f'Failed to read {filepath}: {e}')
    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)

if __name__ == '__main__':

    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    selector_cli = parse_runtime_selector_cli_args()
    overrides = parse_config_overrides(start_index=selector_cli['cursor'])
    cfg = load_config(**overrides)

    data_alias_enriched_hex = selector_cli['data_alias_enriched_hex']
    if not data_alias_enriched_hex:
        raise ValueError('data_alias_enriched_hex is required. Pass it as a positional CLI argument.')
    eval_cfg = cfg['metrics']

    aggregated_metrics_csv = eval_cfg['aggregated_metrics_csv'].replace('#', data_alias_enriched_hex)
    metrics_csv = eval_cfg['all_metrics_csv'].replace('#', data_alias_enriched_hex)
    
    # Strip glob wildcards to get the base output directory.
    data_dir = os.path.dirname(aggregated_metrics_csv)

    aggregated_metrics_csvs = _exclude_merged_outputs(glob.glob(aggregated_metrics_csv))
    metrics_csvs = _exclude_merged_outputs(glob.glob(metrics_csv))

    models_root_dir = eval_cfg['models_dir']

    aggregated_df = merge_metrics(aggregated_metrics_csvs, models_root_dir=models_root_dir).sort_values(by=['Precision', 'Recall', 'F-measure'], ascending=False)
    metrics_df = merge_metrics(metrics_csvs, models_root_dir=models_root_dir).sort_values(by=['Precision', 'Recall', 'F-measure'], ascending=False)

    aggregated_df.to_csv(os.path.join(data_dir, 'aggregated_metrics_merged.csv'), index=False)
    metrics_df.to_csv(os.path.join(data_dir, 'metrics_merged.csv'), index=False)

