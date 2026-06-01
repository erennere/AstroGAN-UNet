import os
import glob
import logging
import pandas as pd

from starter import load_config, parse_config_overrides, _decode_models_dir
from src.evaluation.metrics import parse_runtime_selector_cli_args

def merge_metrics(filepaths):
    results = []
    for filepath in filepaths:
        try:
            df = pd.read_csv(filepath)
            if 'model' not in df.columns:
                logging.warning(f"'model' column not found in {filepath}. Skipping model decode.")
            else:
                model_dicts = df['model'].map(_decode_models_dir)
                df = pd.concat([df, pd.DataFrame(model_dicts.tolist(), index=df.index)], axis=1)
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

    aggregated_metrics_csvs = glob.glob(aggregated_metrics_csv)
    metrics_csvs = glob.glob(metrics_csv)

    aggregated_df = merge_metrics(aggregated_metrics_csvs).sort_values(by=['Precision', 'Recall', 'F-measure'], ascending=False)
    metrics_df = merge_metrics(metrics_csvs).sort_values(by=['Precision', 'Recall', 'F-measure'], ascending=False)

    aggregated_df.to_csv(os.path.join(data_dir, 'aggregated_metrics_merged.csv'), index=False)
    metrics_df.to_csv(os.path.join(data_dir, 'metrics_merged.csv'), index=False)

