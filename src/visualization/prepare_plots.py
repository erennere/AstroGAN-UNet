"""Generate photometric summary tables and hexbin diagnostics from catalogs."""

import os, logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import Normalize
from scipy.stats import binned_statistic
from src.training.utils import ensure_parent_dir_exists
from starter import load_config, parse_config_overrides

def safe_divide(numerator, denominator):
    """Safely divide arrays/scalars while converting zero denominators to NaN."""

    if np.isscalar(denominator):
        return np.nan if denominator == 0 else numerator / denominator
    if isinstance(denominator, pd.Series):
        return numerator / denominator.replace(0, np.nan)
    return numerator / np.where(np.asarray(denominator) == 0, np.nan, denominator)

def finite_mask(df, *columns):
    """Return mask where all requested columns are finite and not null."""

    mask = pd.Series(True, index=df.index)
    for column in columns:
        mask &= df[column].notna() & np.isfinite(df[column])
    return mask

def positive_mask(df, *columns):
    """Return mask where requested columns are finite and strictly positive."""

    mask = finite_mask(df, *columns)
    for column in columns:
        mask &= df[column] > 0
    return mask

def add_summary_metrics(summary_df):
    """Compute precision/recall/F-measure and related derived summary metrics."""

    summary_df['Precision'] = safe_divide(summary_df['TP'], summary_df['TP'] + summary_df['FP'])
    summary_df['Recall'] = safe_divide(summary_df['TP'], summary_df['TP'] + summary_df['FN'])
    summary_df['F-measure'] = safe_divide(
        2 * (summary_df['Precision'] * summary_df['Recall']),
        summary_df['Precision'] + summary_df['Recall'],
    )
    summary_df['IoU'] = safe_divide(summary_df['IOU_sum'], summary_df['union'])
    summary_df['SNR_org'] = safe_divide(summary_df['SNR_org_sum'], summary_df['TP'])
    summary_df['SNR_rec'] = safe_divide(summary_df['SNR_rec_sum'], summary_df['TP'])
    summary_df['SNR_f'] = safe_divide(summary_df['SNR_org'], summary_df['SNR_rec'])
    summary_df['RFE'] = safe_divide(summary_df['RFE_sum'], summary_df['TP'])
    return summary_df

def create_table(metadata_filepath, results_filepath):
    """Aggregate per-image evaluation results into per-ratio and overall tables."""

    df = pd.read_csv(metadata_filepath)
    my_set = set(df.loc[df['location'].str.contains('eval|test', na=False), 'sci_data_set_name'].unique())

    df = pd.read_csv(results_filepath)
    df['dataset'] = df['image_id'].apply(lambda x: x.split('_')[0])
    df = df[df['dataset'].isin(my_set)]
    df['IOU_sum'] = df['IoU']*df['union']
    df['SNR_org_sum'] = df['SNR_org']*df['TP']
    df['SNR_rec_sum'] = df['SNR_rec']*df['TP']
    df['RFE_sum'] = df['RFE']*df['TP']
    df['exp_ratio'] = safe_divide(df['org_exp_time'], df['new_exp_time'])

    summarized_mosaic = df.groupby(['dataset', 'new_exp_time']).agg({
        'TP': 'sum', 
        'FP': 'sum', 
        'FN': 'sum',  
        'IOU_sum': 'sum',  
        'SNR_org_sum': 'sum', 
        'SNR_rec_sum': 'sum',  
        'union': 'sum',
        'org_exp_time' : 'mean',
        'RFE_sum' : 'sum',
        'PSNR_rec' : 'max',
        'PSNR_noisy' : "max",
        'SSIM_rec' : "mean",
        'SSIM_noisy' : 'mean'
    }).reset_index()

    summarized_mosaic = add_summary_metrics(summarized_mosaic)
    summarized_mosaic['exp_ratio'] = safe_divide(summarized_mosaic['org_exp_time'], summarized_mosaic['new_exp_time'])
    summarized_mosaic['IOU_sum'] = summarized_mosaic['IoU']*summarized_mosaic['union']
    summarized_mosaic['SNR_org_sum'] = summarized_mosaic['SNR_org']*summarized_mosaic['TP']
    summarized_mosaic['SNR_rec_sum'] = summarized_mosaic['SNR_rec']*summarized_mosaic['TP']
    summarized_mosaic['RFE_sum'] = summarized_mosaic['RFE']*summarized_mosaic['TP']
    summarized_mosaic['exp_ratio'] = np.ceil(summarized_mosaic['exp_ratio'])
    summarized_mosaic = summarized_mosaic.dropna(subset=['exp_ratio']).copy()
    summarized_mosaic['exp_ratio'] = summarized_mosaic['exp_ratio'].astype(int)

    summarized = summarized_mosaic.groupby(['exp_ratio']).agg({
        'TP': 'sum', 
        'FP': 'sum', 
        'FN': 'sum',  
        'IOU_sum': 'sum',  
        'SNR_org_sum': 'sum', 
        'SNR_rec_sum': 'sum',  
        'union': 'sum',#,  
        'org_exp_time' : 'mean',
        'new_exp_time' : 'mean',
        'RFE_sum' : 'sum',
        'PSNR_rec' : 'mean',
        'PSNR_noisy' : "mean",
        'SSIM_rec' : "mean",
        'SSIM_noisy' : 'mean'
    }).reset_index()
    summarized = add_summary_metrics(summarized)

    part_1 = summarized.drop(labels=['IOU_sum', 'SNR_org_sum', 'SNR_rec_sum', 'union', 'RFE_sum'], axis=1).sort_values(by=['exp_ratio'], ascending=False).reset_index(drop=True)

    summarized_mosaic['temp'] = 1
    summarized = summarized_mosaic.groupby(['temp']).agg({
        'TP': 'sum', 
        'FP': 'sum', 
        'FN': 'sum',  
        'IOU_sum': 'sum',  
        'SNR_org_sum': 'sum', 
        'SNR_rec_sum': 'sum',  
        'union': 'sum',#,  
        'org_exp_time' : 'mean',
        'new_exp_time' : 'mean',
        'RFE_sum' : 'sum',
        'PSNR_rec' : 'mean',
        'PSNR_noisy' : "mean",
        'SSIM_rec' : "mean",
        'SSIM_noisy' : 'mean'
    }).reset_index()

    summarized = add_summary_metrics(summarized)
    part_2 = summarized.drop(labels=['IOU_sum', 'SNR_org_sum', 'SNR_rec_sum', 'union', 'RFE_sum', 'temp'], axis=1)
    part_2['exp_ratio'] = 0
    final_table = pd.concat([part_1, part_2], ignore_index=True).sort_values(by=['exp_ratio'], ascending=False).reset_index(drop=True)
    return final_table

def convert_to_jansky(flux_e_per_s, PHOTPLAM=15369.17570896557, 
                      PHOTFLAM =1.92756031304868e-20, factor=33356.4):
    """Convert flux from e-/s to Jansky using HST calibration constants."""

    return flux_e_per_s*PHOTFLAM*PHOTPLAM**2*factor

def convert_to_angstrom(flux_e_per_s, PHOTFLAM =1.92756031304868e-20):
    """Convert flux from e-/s to cgs per-Angstrom units via PHOTFLAM."""

    return PHOTFLAM*flux_e_per_s

def calculate_abmag(flux_jansky):
    """Convert Jansky flux to AB magnitude, returning NaN for invalid flux."""

    if pd.isna(flux_jansky) or flux_jansky <= 0:
        return np.nan
    return -2.5*np.log10(flux_jansky) + 8.9

def add_columns(df, jansky_factor=None):
    """Add derived ratio, flux-unit, magnitude, and SNR-like columns."""

    df['exp_ratio'] = df.apply(
        lambda row: row['exp_time_rec'] / row['new_exp_time_rec']
        if not pd.isna(row['exp_time_rec']) else
        (row['exp_time_noise'] / row['new_exp_time_noise'] if not pd.isna(row['exp_time_noise']) else 0.0),
        axis=1
    )
    df['exp_ratio'] = df['exp_ratio'].round(2)

    for col in ['flux_x_org', 'flux_y_org', 'flux_x_rec', 'flux_y_rec', 'flux_x_noise', 'flux_y_noise', 'cflux_org', 'cflux_rec', 'cflux_noise',
                'flux_err_org', 'flux_err_rec', 'flux_err_noise']:
        if jansky_factor is None:
            df['j_' + col] = df[col].apply(convert_to_jansky)
        else:
            df['j_' + col] = df[col].apply(convert_to_jansky, factor=jansky_factor)
        if 'err' not in col:
            df['abmag_' + col] = df['j_' + col].apply(calculate_abmag)
        df['a_' + col] = df[col].apply(convert_to_angstrom)

    for flux, flux_err in zip(['flux_x_org', 'flux_y_org', 'flux_x_rec', 'flux_y_rec',
                               'flux_x_noise', 'flux_y_noise', 'cflux_org', 'cflux_rec', 'cflux_noise'],
                              ['flux_err_org', 'flux_err_org', 'flux_err_rec', 'flux_err_rec',
                                'flux_err_noise', 'flux_err_noise', 'flux_err_org', 'flux_err_org', 'flux_err_rec']):
        df['delta_' + flux] = safe_divide(df[flux], df[flux_err])
        df['delta_j_' + flux] = safe_divide(df['j_' + flux], df['j_' + flux_err])
        df['delta_a_' + flux] = safe_divide(df['a_' + flux], df['a_' + flux_err])
    return df

def get_exp_ratio_subset(df, exp_ratio, total_label):
    """Return subset/label pair for one exposure-ratio bucket or total."""

    if int(exp_ratio) != 0:
        return df[df['exp_ratio'] == exp_ratio], rf'$\gamma={int(exp_ratio)}$'
    return df, total_label

def get_gridsize(values):
    """Choose a hexbin gridsize scaled to sample count."""

    return max(100, int(np.sqrt(len(values) / 10)))

def build_norm(series, quantiles):
    """Build a robust matplotlib normalization object from quantile bounds."""

    finite_values = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()
    if finite_values.empty:
        raise ValueError('Cannot build normalization from an empty or non-finite series.')

    lower_quantile, upper_quantile = quantiles
    vmin, vmax = np.percentile(finite_values, [lower_quantile, upper_quantile])
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError('Normalization quantiles produced non-finite bounds.')
    if vmin == vmax:
        vmin = finite_values.min()
        vmax = finite_values.max()
    if vmin == vmax:
        raise ValueError('Normalization bounds collapsed to a single value.')

    return Normalize(vmin=vmin, vmax=vmax)

def clip_single_series(x_values, y_values, x_percentiles=(1, 99), y_percentiles=(1, 99)):
    """Clip paired x/y series to percentile windows and return limits."""

    temp_df = pd.DataFrame({'x': x_values, 'y': y_values}).dropna()
    if temp_df.empty:
        return None

    y_low, y_high = np.percentile(temp_df['y'], y_percentiles)
    x_low, x_high = np.percentile(temp_df['x'], x_percentiles)
    temp_df = temp_df[
        (temp_df['y'] >= y_low) & (temp_df['y'] <= y_high)
        & (temp_df['x'] >= x_low) & (temp_df['x'] <= x_high)
    ]
    if temp_df.empty:
        return None
    return temp_df['x'], temp_df['y'], (x_low, x_high), (y_low, y_high)


def clip_aligned_paired_series(x_rec, y_rec, x_noise, y_noise, y_percentiles=(1, 99)):
    """Clip rec/noise series to a shared y-window for aligned comparisons."""

    temp_df = pd.DataFrame(
        {'x_rec': x_rec, 'y_rec': y_rec, 'x_noise': x_noise, 'y_noise': y_noise}
    ).dropna()
    if temp_df.empty:
        return None

    rec_low, rec_high = np.percentile(temp_df['y_rec'], y_percentiles)
    noise_low, noise_high = np.percentile(temp_df['y_noise'], y_percentiles)
    y_low = min(rec_low, noise_low)
    y_high = max(rec_high, noise_high)
    temp_df = temp_df[
        (temp_df['y_rec'] >= y_low) & (temp_df['y_rec'] <= y_high)
        & (temp_df['y_noise'] >= y_low) & (temp_df['y_noise'] <= y_high)
    ]
    if temp_df.empty:
        return None

    return temp_df['x_rec'], temp_df['y_rec'], temp_df['x_noise'], temp_df['y_noise'], (y_low, y_high)

def clip_separate_paired_series(x_rec, y_rec, x_noise, y_noise, y_percentiles=(1, 99)):
    """Clip rec/noise series independently and return filtered values."""

    temp_rec = pd.DataFrame({'x': x_rec, 'y': y_rec}).dropna()
    temp_noise = pd.DataFrame({'x': x_noise, 'y': y_noise}).dropna()
    if temp_rec.empty or temp_noise.empty:
        return None

    rec_low, rec_high = np.percentile(temp_rec['y'], y_percentiles)
    noise_low, noise_high = np.percentile(temp_noise['y'], y_percentiles)
    y_low = min(rec_low, noise_low)
    y_high = max(rec_high, noise_high)

    temp_rec = temp_rec[(temp_rec['y'] >= y_low) & (temp_rec['y'] <= y_high)]
    temp_noise = temp_noise[(temp_noise['y'] >= y_low) & (temp_noise['y'] <= y_high)]
    if temp_rec.empty or temp_noise.empty:
        return None

    return temp_rec['x'], temp_rec['y'], temp_noise['x'], temp_noise['y'], (y_low, y_high)


def plot_hexbin_panel(
    fig,
    ax,
    x_values,
    y_values,
    *,
    cmap_name,
    cmap,
    gridsize,
    xlabel=None,
    ylabel=None,
    title=None,
    xlim=None,
    ylim=None,
    legend_label=None,
    legend_value=None,
    norm=None,
    add_one_to_one=False,
):
    """Render one configured hexbin panel and optional legend/1:1 line."""

    if len(x_values) == 0 or len(y_values) == 0:
        return False

    if xlim is not None:
        ax.set_xlim(list(xlim))
    if ylim is not None:
        ax.set_ylim(list(ylim))

    ax.set_box_aspect(1)
    if add_one_to_one:
        left = np.min(x_values)
        right = np.max(x_values)
        one_to_one = np.linspace(left, right, 1000)
        ax.plot(one_to_one, one_to_one, c='red', linestyle='dashed', alpha=0.5)

    hexbin = ax.hexbin(
        x_values,
        y_values,
        gridsize=gridsize,
        cmap=cmap_name,
        bins='log',
        mincnt=2,
        linewidths=0.1,
    )
    if norm is None:
        fig.colorbar(hexbin, ax=ax)
    else:
        fig.colorbar(hexbin, ax=ax, norm=norm)

    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if legend_label is not None:
        color = cmap(0.5)
        if norm is not None and legend_value is not None and np.isfinite(legend_value):
            color = cmap(norm(legend_value))
        gradient_line = mlines.Line2D([0], [0], color=color, lw=4, label=legend_label)
        ax.legend(
            handles=[gradient_line],
            loc='lower center',
            bbox_to_anchor=(0.5, 1.05),
            ncol=1,
            frameon=False,
        )

    ax.grid(True)
    return True


def finalize_plot(fig, output_filepath, suptitle, dpi):
    """Finalize figure layout/title and save to disk."""

    fig.suptitle(suptitle, fontsize=20)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.2, wspace=0.3, top=0.92)
    fig.savefig(output_filepath, dpi=dpi)
    plt.close(fig)


def build_mask(df, columns, mask_spec):
    """Build a boolean row mask from finite/positive/missing rules."""

    if mask_spec is None:
        return pd.Series(True, index=df.index)

    mask = pd.Series(True, index=df.index)
    positive_columns = [columns[key] for key in mask_spec.get('positive', ())]
    finite_columns = [columns[key] for key in mask_spec.get('finite', ())]

    if finite_columns:
        mask &= finite_mask(df, *finite_columns)
    if positive_columns:
        mask &= positive_mask(df, *positive_columns)

    for key in mask_spec.get('missing', ()):
        mask &= df[columns[key]].isna()

    return mask


def transform_series(df, columns, series_spec):
    """Transform a plotted series using identity/log/ratio/delta modes."""

    values = df[columns[series_spec['key']]]
    transform = series_spec.get('transform', 'identity')

    if transform == 'identity':
        return values
    if transform == 'log10':
        return np.log10(values)
    if transform == 'ratio':
        return safe_divide(values, df[columns[series_spec['reference']]])
    if transform == 'delta':
        return values - df[columns[series_spec['reference']]]

    raise ValueError(f"Unsupported transform: {transform}")


def transform_limits(limits, series_spec):
    """Apply x-axis limit transformation consistent with a series spec."""

    if limits is None:
        return None

    transform = series_spec.get('transform', 'identity')
    if transform == 'identity':
        return limits
    if transform == 'log10':
        return tuple(np.log10(limits))

    raise ValueError(f"Unsupported limit transform: {transform}")


def get_legend_value(values, strategy):
    """Compute legend color anchor value using requested summary strategy."""

    if strategy == 'median':
        return np.median(values)
    if strategy == 'min':
        return values.min()
    return None


def build_panel_norm(df, columns, spec, norm_quantiles, x_values=None, y_values=None):
    """Construct a panel normalization either from fixed column or dynamic data."""

    norm_mode = spec.get('norm_mode')
    if norm_mode == 'column':
        return build_norm(df[columns[spec['norm_key']]], norm_quantiles)
    if norm_mode == 'dynamic' and x_values is not None and y_values is not None:
        return build_norm(pd.concat([x_values, y_values], ignore_index=True), norm_quantiles)
    return None


def apply_pre_window(sub_df, columns, window_spec):
    """Apply optional percentile-based pre-windowing to a panel subset."""

    if window_spec is None:
        return sub_df, None

    column = columns[window_spec['key']]
    if window_spec.get('positive'):
        mask = positive_mask(sub_df, column)
    else:
        mask = finite_mask(sub_df, column)

    if not mask.any():
        return None, None

    low, high = np.percentile(sub_df.loc[mask, column], window_spec['percentiles'])
    return sub_df[(sub_df[column] >= low) & (sub_df[column] <= high)], (low, high)


def prepare_single_panel(sub_df, spec, columns, norm_quantiles):
    """Prepare clipped data arrays and norm for one single-panel plot."""

    mask = build_mask(sub_df, columns, spec['mask'])
    if not mask.any():
        return None

    panel_df = sub_df.loc[mask]
    x_values = transform_series(panel_df, columns, spec['x'])
    y_values = transform_series(panel_df, columns, spec['y'])
    clipped = clip_single_series(
        x_values,
        y_values,
        x_percentiles=spec.get('x_clip', (1, 99)),
        y_percentiles=spec.get('y_clip', (1, 99)),
    )
    if clipped is None:
        return None

    x_values, y_values, _, _ = clipped
    xlim = None
    if spec.get('xlim_floor') is not None:
        xlim = (spec['xlim_floor'], x_values.max())

    return {
        'x': x_values,
        'y': y_values,
        'xlim': xlim,
        'norm': build_panel_norm(sub_df, columns, spec, norm_quantiles, x_values, y_values),
    }


def prepare_paired_panel(sub_df, spec, columns):
    """Prepare aligned/separate rec-noise arrays and limits for paired panels."""

    sub_df, raw_limits = apply_pre_window(sub_df, columns, spec.get('pre_window'))
    if sub_df is None or sub_df.empty:
        return None

    if spec['mask_mode'] == 'shared':
        shared_mask = build_mask(sub_df, columns, spec['mask'])
        if not shared_mask.any():
            return None
        rec_df = sub_df.loc[shared_mask]
        noise_df = sub_df.loc[shared_mask]
    else:
        rec_mask = build_mask(sub_df, columns, spec['rec_mask'])
        noise_mask = build_mask(sub_df, columns, spec['noise_mask'])
        if not rec_mask.any() or not noise_mask.any():
            return None
        rec_df = sub_df.loc[rec_mask]
        noise_df = sub_df.loc[noise_mask]

    x_rec = transform_series(rec_df, columns, spec['rec']['x'])
    y_rec = transform_series(rec_df, columns, spec['rec']['y'])
    x_noise = transform_series(noise_df, columns, spec['noise']['x'])
    y_noise = transform_series(noise_df, columns, spec['noise']['y'])

    clip_fn = clip_aligned_paired_series if spec['clip_mode'] == 'aligned' else clip_separate_paired_series
    clipped = clip_fn(
        x_rec,
        y_rec,
        x_noise,
        y_noise,
        y_percentiles=spec.get('y_clip', (1, 99)),
    )
    if clipped is None:
        return None

    x_rec, y_rec, x_noise, y_noise, y_limits = clipped
    x_limits = transform_limits(raw_limits, spec['rec']['x'])
    return {
        'x_rec': x_rec,
        'y_rec': y_rec,
        'x_noise': x_noise,
        'y_noise': y_noise,
        'x_limits': x_limits,
        'y_limits': y_limits,
    }


def render_single_hexbin_grid(df, output_filepath, spec, columns, norm_quantiles):
    """Render a 2x4 grid of single-series hexbin panels by exp_ratio."""

    ensure_parent_dir_exists(output_filepath)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    cmap = plt.get_cmap(spec['rec_cmap'])
    base_norm = build_panel_norm(df, columns, spec, norm_quantiles)

    for index, exp_ratio in enumerate(np.sort(df['exp_ratio'].unique())):
        column = index // 2
        if column >= axes.shape[1]:
            logging.warning('Single-panel grid supports at most 8 exp_ratio groups. Stopping at exp_ratio %s.', exp_ratio)
            break

        row = index % 2
        sub_df, label = get_exp_ratio_subset(df, exp_ratio, spec.get('total_label', 'Total'))
        panel_data = prepare_single_panel(sub_df, spec, columns, norm_quantiles)
        if panel_data is None:
            logging.warning('No plot data for %s at exp_ratio %s, skipping.', spec['output_filename'], exp_ratio)
            continue

        plot_hexbin_panel(
            fig,
            axes[row, column],
            panel_data['x'],
            panel_data['y'],
            cmap_name=spec['rec_cmap'],
            cmap=cmap,
            gridsize=get_gridsize(panel_data['x']),
            xlabel=spec['xlabel'] if row == 1 else None,
            ylabel=spec['ylabel'] if column == 0 else None,
            xlim=panel_data['xlim'],
            legend_label=f"{label}, N={len(panel_data['x'])}",
            legend_value=get_legend_value(panel_data['x'], spec.get('legend_stat', 'min')),
            norm=panel_data['norm'] if panel_data['norm'] is not None else base_norm,
            add_one_to_one=spec.get('one_to_one', False),
        )

    finalize_plot(fig, output_filepath, spec['suptitle'], spec.get('dpi', 250))


def render_paired_hexbin_grid(df, output_filepath, spec, columns, norm_quantiles):
    """Render a 4x4 paired rec/noise hexbin grid by exp_ratio buckets."""

    ensure_parent_dir_exists(output_filepath)
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    rec_cmap = plt.get_cmap(spec['rec_cmap'])
    noise_cmap = plt.get_cmap(spec['noise_cmap'])
    base_norm = build_panel_norm(df, columns, spec, norm_quantiles)

    for index, exp_ratio in enumerate(np.sort(df['exp_ratio'].unique())):
        if index >= 8:
            logging.warning('Paired grid supports at most 8 exp_ratio groups. Stopping at exp_ratio %s.', exp_ratio)
            break

        row = index % 4
        rec_column, noise_column = (0, 1) if index < 4 else (2, 3)
        sub_df, label = get_exp_ratio_subset(df, exp_ratio, spec.get('total_label', r"all $\gamma$'s"))
        panel_data = prepare_paired_panel(sub_df, spec, columns)
        if panel_data is None:
            logging.warning('No plot data for %s at exp_ratio %s, skipping.', spec['output_filename'], exp_ratio)
            continue

        plot_hexbin_panel(
            fig,
            axes[row, rec_column],
            panel_data['x_rec'],
            panel_data['y_rec'],
            cmap_name=spec['rec_cmap'],
            cmap=rec_cmap,
            gridsize=get_gridsize(panel_data['x_rec']),
            xlabel=spec.get('xlabel_rec') if row == 3 else None,
            ylabel=spec.get('ylabel_rec'),
            title='Reconstructed Image' if row == 0 else None,
            xlim=panel_data['x_limits'],
            ylim=panel_data['y_limits'],
            legend_label=f"{label}, N={len(panel_data['x_rec'])}",
            legend_value=get_legend_value(panel_data['x_rec'], spec.get('legend_stat', 'median')),
            norm=base_norm,
            add_one_to_one=spec.get('one_to_one', False),
        )
        plot_hexbin_panel(
            fig,
            axes[row, noise_column],
            panel_data['x_noise'],
            panel_data['y_noise'],
            cmap_name=spec['noise_cmap'],
            cmap=noise_cmap,
            gridsize=get_gridsize(panel_data['x_noise']),
            xlabel=spec.get('xlabel_noise') if row == 3 else None,
            ylabel=spec.get('ylabel_noise'),
            title='Noisy Image' if row == 0 else None,
            xlim=panel_data['x_limits'],
            ylim=panel_data['y_limits'],
            legend_label=f"{label}, N={len(panel_data['x_noise'])}",
            legend_value=get_legend_value(panel_data['x_noise'], spec.get('legend_stat', 'median')),
            norm=base_norm,
            add_one_to_one=spec.get('one_to_one', False),
        )

    finalize_plot(fig, output_filepath, spec['suptitle'], spec.get('dpi', 500))


def render_hexbin_plot(df, output_filepath, spec, columns, norm_quantiles):
    """Dispatch to single or paired hexbin layout renderer."""

    if spec['layout'] == 'single':
        render_single_hexbin_grid(df, output_filepath, spec, columns, norm_quantiles)
        return
    if spec['layout'] == 'paired':
        render_paired_hexbin_grid(df, output_filepath, spec, columns, norm_quantiles)
        return
    raise ValueError(f"Unsupported plot layout: {spec['layout']}")


def build_hexbin_plot_specs(rec_cmap, noise_cmap):
    """Return declarative plot specifications for all hexbin outputs."""

    flux_label = r'Org. Flux $[erg\ s^{-1}\ cm^{-2}\ \AA^{-1}]$'
    rec_flux_label = r'Rec. Flux $[erg\ s^{-1}\ cm^{-2}\ \AA^{-1}]$'
    noisy_flux_label = r'Noisy Flux $[erg\ s^{-1}\ cm^{-2}\ \AA^{-1}]$'
    kron_label = r'Org. Kron Radius $[arcsec]$'
    arcsec_label = r'$[arcsec]$'
    abmag_label = r'AB MAG'
    delta_kron_label = r'$\Delta$Kron Radius $[arcsec]$'

    return [
        {
            'layout': 'paired',
            'output_filename': 'flux_ratio_combined_both.png',
            'suptitle': r"Flux Ratio vs. $\log$(Flux) (Detected in Both)",
            'rec_cmap': rec_cmap,
            'noise_cmap': noise_cmap,
            'pre_window': {'key': 'org_flux', 'positive': True, 'percentiles': (1, 98)},
            'mask_mode': 'shared',
            'mask': {'positive': ['org_flux'], 'finite': ['rec_flux', 'noise_flux']},
            'rec': {'x': {'key': 'org_flux', 'transform': 'log10'}, 'y': {'key': 'rec_flux', 'transform': 'ratio', 'reference': 'org_flux'}},
            'noise': {'x': {'key': 'org_flux', 'transform': 'log10'}, 'y': {'key': 'noise_flux', 'transform': 'ratio', 'reference': 'org_flux'}},
            'clip_mode': 'aligned',
            'norm_mode': 'column',
            'norm_key': 'org_flux',
            'legend_stat': 'median',
            'xlabel_rec': flux_label,
            'xlabel_noise': flux_label,
            'ylabel_rec': r'Flux Ratio',
        },
        {
            'layout': 'paired',
            'output_filename': 'delta_kron_combined_both.png',
            'suptitle': r"Rec., Noisy $\Delta$Kron Radius vs. Org. Kron Radius (Detected in Both)",
            'rec_cmap': rec_cmap,
            'noise_cmap': noise_cmap,
            'pre_window': {'key': 'org_kron', 'positive': False, 'percentiles': (1, 98)},
            'mask_mode': 'shared',
            'mask': {'finite': ['org_kron', 'rec_kron', 'noise_kron']},
            'rec': {'x': {'key': 'org_kron'}, 'y': {'key': 'rec_kron', 'transform': 'delta', 'reference': 'org_kron'}},
            'noise': {'x': {'key': 'org_kron'}, 'y': {'key': 'noise_kron', 'transform': 'delta', 'reference': 'org_kron'}},
            'clip_mode': 'aligned',
            'norm_mode': 'column',
            'norm_key': 'org_kron',
            'legend_stat': 'min',
            'xlabel_rec': r'Kron Radius $[arcsec]$',
            'xlabel_noise': arcsec_label,
            'ylabel_rec': delta_kron_label,
        },
        {
            'layout': 'paired',
            'output_filename': 'flux_ratio_mag_combined_both.png',
            'suptitle': r"Flux Ratio vs. AB MAG (Detected in Both)",
            'rec_cmap': rec_cmap,
            'noise_cmap': noise_cmap,
            'pre_window': {'key': 'org_mag', 'positive': False, 'percentiles': (1, 98)},
            'mask_mode': 'shared',
            'mask': {'positive': ['org_flux'], 'finite': ['rec_flux', 'noise_flux', 'org_mag']},
            'rec': {'x': {'key': 'org_mag'}, 'y': {'key': 'rec_flux', 'transform': 'ratio', 'reference': 'org_flux'}},
            'noise': {'x': {'key': 'org_mag'}, 'y': {'key': 'noise_flux', 'transform': 'ratio', 'reference': 'org_flux'}},
            'clip_mode': 'aligned',
            'norm_mode': 'column',
            'norm_key': 'org_flux',
            'legend_stat': 'median',
            'xlabel_rec': abmag_label,
            'xlabel_noise': abmag_label,
            'ylabel_rec': r'Flux Ratio',
        },
        {
            'layout': 'paired',
            'output_filename': 'delta_kron_mag_combined_both.png',
            'suptitle': r"Rec., Noisy $\Delta$Kron Radius vs. AB MAG (Detected in Both)",
            'rec_cmap': rec_cmap,
            'noise_cmap': noise_cmap,
            'pre_window': {'key': 'org_mag', 'positive': False, 'percentiles': (1, 98)},
            'mask_mode': 'shared',
            'mask': {'finite': ['org_kron', 'rec_kron', 'noise_kron', 'org_mag']},
            'rec': {'x': {'key': 'org_mag'}, 'y': {'key': 'rec_kron', 'transform': 'delta', 'reference': 'org_kron'}},
            'noise': {'x': {'key': 'org_mag'}, 'y': {'key': 'noise_kron', 'transform': 'delta', 'reference': 'org_kron'}},
            'clip_mode': 'aligned',
            'norm_mode': 'column',
            'norm_key': 'org_kron',
            'legend_stat': 'min',
            'xlabel_rec': abmag_label,
            'xlabel_noise': abmag_label,
            'ylabel_rec': delta_kron_label,
        },
        {
            'layout': 'single',
            'output_filename': 'delta_kron_mag_rec.png',
            'suptitle': r"Rec. $\Delta$Kron Radius vs. AB MAG (Detected only in Rec.)",
            'rec_cmap': rec_cmap,
            'mask': {'finite': ['org_kron', 'rec_kron', 'org_mag'], 'missing': ['noise_kron']},
            'x': {'key': 'org_mag'},
            'y': {'key': 'rec_kron', 'transform': 'delta', 'reference': 'org_kron'},
            'norm_mode': 'dynamic',
            'legend_stat': 'min',
            'xlim_floor': 22.7,
            'xlabel': abmag_label,
            'ylabel': delta_kron_label,
        },
        {
            'layout': 'single',
            'output_filename': 'flux_ratio_mag_rec.png',
            'suptitle': r"Rec. Flux Ratio vs. AB MAG (Detected only in Rec.)",
            'rec_cmap': rec_cmap,
            'mask': {'positive': ['org_flux'], 'finite': ['rec_flux', 'org_mag'], 'missing': ['noise_flux']},
            'x': {'key': 'org_mag'},
            'y': {'key': 'rec_flux', 'transform': 'ratio', 'reference': 'org_flux'},
            'norm_mode': 'dynamic',
            'legend_stat': 'min',
            'xlim_floor': 22.7,
            'xlabel': abmag_label,
            'ylabel': r'Flux Ratio',
        },
        {
            'layout': 'single',
            'output_filename': 'delta_kron_rec.png',
            'suptitle': r"Rec. $\Delta$Kron Radius vs. Org. Kron Radius (Detected only in Rec.)",
            'rec_cmap': rec_cmap,
            'mask': {'finite': ['org_kron', 'rec_kron'], 'missing': ['noise_kron']},
            'x': {'key': 'org_kron'},
            'y': {'key': 'rec_kron', 'transform': 'delta', 'reference': 'org_kron'},
            'norm_mode': 'dynamic',
            'legend_stat': 'min',
            'xlabel': kron_label,
            'ylabel': delta_kron_label,
        },
        {
            'layout': 'single',
            'output_filename': 'flux_ratio_rec.png',
            'suptitle': r"Rec.Flux Ratio vs. Org. $\log($Flux) (Detected only in Rec.)",
            'rec_cmap': rec_cmap,
            'mask': {'positive': ['org_flux'], 'finite': ['rec_flux'], 'missing': ['noise_flux']},
            'x': {'key': 'org_flux', 'transform': 'log10'},
            'y': {'key': 'rec_flux', 'transform': 'ratio', 'reference': 'org_flux'},
            'norm_mode': 'column',
            'norm_key': 'org_flux',
            'legend_stat': 'min',
            'xlabel': flux_label,
            'ylabel': r'Rec. Flux Ratio',
        },
        {
            'layout': 'paired',
            'output_filename': 'kron_combined.png',
            'suptitle': r"Rec., Noisy Kron Radius vs. Org. Kron Radius",
            'rec_cmap': rec_cmap,
            'noise_cmap': noise_cmap,
            'pre_window': {'key': 'org_kron', 'positive': False, 'percentiles': (1, 98)},
            'mask_mode': 'separate',
            'rec_mask': {'finite': ['org_kron', 'rec_kron']},
            'noise_mask': {'finite': ['org_kron', 'noise_kron']},
            'rec': {'x': {'key': 'org_kron'}, 'y': {'key': 'rec_kron'}},
            'noise': {'x': {'key': 'org_kron'}, 'y': {'key': 'noise_kron'}},
            'clip_mode': 'separate',
            'norm_mode': 'column',
            'norm_key': 'org_kron',
            'legend_stat': 'min',
            'xlabel_rec': kron_label,
            'xlabel_noise': arcsec_label,
            'ylabel_rec': r'Kron Radius $[arcsec]$',
            'one_to_one': True,
        },
        {
            'layout': 'paired',
            'output_filename': 'flux_combined.png',
            'suptitle': r"Rec., Noisy $\log($Flux) vs. Org. $\log($Flux)",
            'rec_cmap': rec_cmap,
            'noise_cmap': noise_cmap,
            'pre_window': {'key': 'org_flux', 'positive': True, 'percentiles': (1, 98)},
            'mask_mode': 'separate',
            'rec_mask': {'positive': ['org_flux', 'rec_flux']},
            'noise_mask': {'positive': ['org_flux', 'noise_flux']},
            'rec': {'x': {'key': 'org_flux', 'transform': 'log10'}, 'y': {'key': 'rec_flux', 'transform': 'log10'}},
            'noise': {'x': {'key': 'org_flux', 'transform': 'log10'}, 'y': {'key': 'noise_flux', 'transform': 'log10'}},
            'clip_mode': 'separate',
            'norm_mode': 'column',
            'norm_key': 'org_flux',
            'legend_stat': 'median',
            'xlabel_rec': flux_label,
            'xlabel_noise': flux_label,
            'ylabel_rec': rec_flux_label,
            'ylabel_noise': noisy_flux_label,
            'one_to_one': True,
        },
        {
            'layout': 'paired',
            'output_filename': 'kron_combined_both.png',
            'suptitle': r"Rec., Noisy Kron Radius vs. Org. Kron Radius (Detected in Both)",
            'rec_cmap': rec_cmap,
            'noise_cmap': noise_cmap,
            'pre_window': {'key': 'org_kron', 'positive': False, 'percentiles': (1, 98)},
            'mask_mode': 'shared',
            'mask': {'finite': ['org_kron', 'rec_kron', 'noise_kron']},
            'rec': {'x': {'key': 'org_kron'}, 'y': {'key': 'rec_kron'}},
            'noise': {'x': {'key': 'org_kron'}, 'y': {'key': 'noise_kron'}},
            'clip_mode': 'aligned',
            'norm_mode': 'column',
            'norm_key': 'org_kron',
            'legend_stat': 'min',
            'xlabel_rec': kron_label,
            'xlabel_noise': arcsec_label,
            'ylabel_rec': r'Kron Radius $[arcsec]$',
            'one_to_one': True,
        },
        {
            'layout': 'paired',
            'output_filename': 'flux_combined_both.png',
            'suptitle': r"Rec., Noisy $\log($Flux) vs. Org. $\log($Flux) (Detected in Both)",
            'rec_cmap': rec_cmap,
            'noise_cmap': noise_cmap,
            'pre_window': {'key': 'org_flux', 'positive': True, 'percentiles': (1, 98)},
            'mask_mode': 'shared',
            'mask': {'positive': ['org_flux', 'rec_flux', 'noise_flux']},
            'rec': {'x': {'key': 'org_flux', 'transform': 'log10'}, 'y': {'key': 'rec_flux', 'transform': 'log10'}},
            'noise': {'x': {'key': 'org_flux', 'transform': 'log10'}, 'y': {'key': 'noise_flux', 'transform': 'log10'}},
            'clip_mode': 'aligned',
            'norm_mode': 'column',
            'norm_key': 'org_flux',
            'legend_stat': 'median',
            'xlabel_rec': flux_label,
            'xlabel_noise': flux_label,
            'ylabel_rec': rec_flux_label,
            'ylabel_noise': noisy_flux_label,
            'one_to_one': True,
        },
        {
            'layout': 'single',
            'output_filename': 'flux_rec.png',
            'suptitle': r"Rec. $\log($Flux) vs. Org. $\log($Flux) (Detected only in Rec.)",
            'rec_cmap': rec_cmap,
            'mask': {'finite': ['org_flux', 'rec_flux'], 'missing': ['noise_flux']},
            'x': {'key': 'org_flux'},
            'y': {'key': 'rec_flux'},
            'norm_mode': 'column',
            'norm_key': 'org_flux',
            'legend_stat': 'min',
            'xlabel': flux_label,
            'ylabel': rec_flux_label,
            'one_to_one': True,
        },
        {
            'layout': 'single',
            'output_filename': 'kron_rec.png',
            'suptitle': r"Rec. Kron Radius vs. Org. Kron Radius (Detected only in Rec.)",
            'rec_cmap': rec_cmap,
            'mask': {'finite': ['org_kron', 'rec_kron'], 'missing': ['noise_kron']},
            'x': {'key': 'org_kron'},
            'y': {'key': 'rec_kron'},
            'y_clip': (1, 98),
            'norm_mode': 'dynamic',
            'legend_stat': 'min',
            'xlabel': kron_label,
            'ylabel': r'Rec. Kron Radius $[arcsec]$',
            'one_to_one': True,
        },
        {
            'layout': 'paired',
            'output_filename': 'flux_ratio_mag_combined.png',
            'suptitle': r"Flux Ratio vs. AB MAG",
            'rec_cmap': rec_cmap,
            'noise_cmap': noise_cmap,
            'pre_window': {'key': 'org_mag', 'positive': False, 'percentiles': (1, 98)},
            'mask_mode': 'separate',
            'rec_mask': {'positive': ['org_flux'], 'finite': ['rec_flux', 'org_mag']},
            'noise_mask': {'positive': ['org_flux'], 'finite': ['noise_flux', 'org_mag']},
            'rec': {'x': {'key': 'org_mag'}, 'y': {'key': 'rec_flux', 'transform': 'ratio', 'reference': 'org_flux'}},
            'noise': {'x': {'key': 'org_mag'}, 'y': {'key': 'noise_flux', 'transform': 'ratio', 'reference': 'org_flux'}},
            'clip_mode': 'separate',
            'norm_mode': 'column',
            'norm_key': 'org_flux',
            'legend_stat': 'median',
            'xlabel_rec': abmag_label,
            'xlabel_noise': abmag_label,
            'ylabel_rec': r'Flux Ratio',
        },
        {
            'layout': 'paired',
            'output_filename': 'delta_kron_mag_combined.png',
            'suptitle': r"Rec., Noisy $\Delta$Kron Radius vs. AB MAG",
            'rec_cmap': rec_cmap,
            'noise_cmap': noise_cmap,
            'pre_window': {'key': 'org_mag', 'positive': False, 'percentiles': (1, 98)},
            'mask_mode': 'separate',
            'rec_mask': {'finite': ['org_kron', 'rec_kron', 'org_mag']},
            'noise_mask': {'finite': ['org_kron', 'noise_kron', 'org_mag']},
            'rec': {'x': {'key': 'org_mag'}, 'y': {'key': 'rec_kron', 'transform': 'delta', 'reference': 'org_kron'}},
            'noise': {'x': {'key': 'org_mag'}, 'y': {'key': 'noise_kron', 'transform': 'delta', 'reference': 'org_kron'}},
            'clip_mode': 'separate',
            'norm_mode': 'column',
            'norm_key': 'org_kron',
            'legend_stat': 'min',
            'xlabel_rec': abmag_label,
            'xlabel_noise': abmag_label,
            'ylabel_rec': delta_kron_label,
        },
        {
            'layout': 'paired',
            'output_filename': 'flux_ratio_combined.png',
            'suptitle': r"Flux Ratio vs. $\log$(Flux)",
            'rec_cmap': rec_cmap,
            'noise_cmap': noise_cmap,
            'pre_window': {'key': 'org_flux', 'positive': True, 'percentiles': (1, 98)},
            'mask_mode': 'separate',
            'rec_mask': {'positive': ['org_flux'], 'finite': ['rec_flux']},
            'noise_mask': {'positive': ['org_flux'], 'finite': ['noise_flux']},
            'rec': {'x': {'key': 'org_flux', 'transform': 'log10'}, 'y': {'key': 'rec_flux', 'transform': 'ratio', 'reference': 'org_flux'}},
            'noise': {'x': {'key': 'org_flux', 'transform': 'log10'}, 'y': {'key': 'noise_flux', 'transform': 'ratio', 'reference': 'org_flux'}},
            'clip_mode': 'separate',
            'norm_mode': 'column',
            'norm_key': 'org_flux',
            'legend_stat': 'median',
            'xlabel_rec': flux_label,
            'xlabel_noise': flux_label,
            'ylabel_rec': r'Flux Ratio',
        },
        {
            'layout': 'paired',
            'output_filename': 'delta_kron_combined.png',
            'suptitle': r"Rec., Noisy $\Delta$Kron Radius vs. Org. Kron Radius",
            'rec_cmap': rec_cmap,
            'noise_cmap': noise_cmap,
            'pre_window': {'key': 'org_kron', 'positive': False, 'percentiles': (1, 98)},
            'mask_mode': 'separate',
            'rec_mask': {'finite': ['org_kron', 'rec_kron']},
            'noise_mask': {'finite': ['org_kron', 'noise_kron']},
            'rec': {'x': {'key': 'org_kron'}, 'y': {'key': 'rec_kron', 'transform': 'delta', 'reference': 'org_kron'}},
            'noise': {'x': {'key': 'org_kron'}, 'y': {'key': 'noise_kron', 'transform': 'delta', 'reference': 'org_kron'}},
            'clip_mode': 'separate',
            'norm_mode': 'column',
            'norm_key': 'org_kron',
            'legend_stat': 'min',
            'xlabel_rec': r'Kron Radius $[arcsec]$',
            'xlabel_noise': arcsec_label,
            'ylabel_rec': delta_kron_label,
        },
    ]


def binned_median(x, y, bins):
    """Compute median and IQR statistics of y within x-bins."""

    # Convert pd.NA to np.nan
    x, y = x.astype(float), y.astype(float)
    # Remove NaNs
    valid = ~(np.isnan(x) | np.isnan(y)) & np.isfinite(x) & np.isfinite(y) & (x > 0)
    x, y = x[valid], y[valid]

    # Compute binned statistics
    bin_means, bin_edges, _ = binned_statistic(x, y, statistic='median', bins=bins)
    bin_p25, _, _ = binned_statistic(x, y, statistic=lambda y: np.nanpercentile(y, 25), bins=bins)
    bin_p75, _, _ = binned_statistic(x, y, statistic=lambda y: np.nanpercentile(y, 75), bins=bins)
    # Bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # Remove NaN bins
    valid_bins = ~np.isnan(bin_means)
    return bin_centers[valid_bins], bin_means[valid_bins], bin_p25[valid_bins], bin_p75[valid_bins]

def new_metrics(df, exp_time_col='exp_ratio'):
    """Compute TP/FP/FN precision-recall metrics grouped by exposure ratio."""

    results = []
    for exp_ratio, group in df.groupby(exp_time_col):
        tp = group[(~group['flux_x_org'].isna()) & (~group['flux_x_rec'].isna())]
        fp = group[(group['flux_x_org'].isna()) & (~group['flux_x_rec'].isna())]
        fn = group[(~group['flux_x_org'].isna()) & (group['flux_x_rec'].isna())]
        
        tp_count, fp_count, fn_count = len(tp), len(fp), len(fn)
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
        f_measure = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results.append({
            'exp_ratio': int(exp_ratio),
            'TP': tp_count,
            'FP': fp_count,
            'FN': fn_count,
            'Precision': precision,
            'Recall': recall,
            'F-measure': f_measure
        })
    
    summary_df = pd.DataFrame(results)
    
    # Compute overall statistics
    overall_tp = summary_df['TP'].sum()
    overall_fp = summary_df['FP'].sum()
    overall_fn = summary_df['FN'].sum()
    overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
    overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
    overall_f_measure = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    overall_stats = pd.DataFrame([{
        'exp_ratio': 0,
        'TP': overall_tp,
        'FP': overall_fp,
        'FN': overall_fn,
        'Precision': overall_precision,
        'Recall': overall_recall,
        'F-measure': overall_f_measure
    }])
    return pd.concat([summary_df, overall_stats], ignore_index=True)

def create_flux_flux_error_diagram(df, flux_org_col, flux_noise_col, flux_rec_col, flux_org_err_col,
                                    flux_noise_err_col, flux_rec_err_col, output_filepath):
    """Create SNR-vs-flux binned summary plots for org/noisy/reconstructed data."""

    ensure_parent_dir_exists(output_filepath)

    # Create a 2x4 subplot grid
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    j = 0
    df = df.copy()
    df['nsr_org'] = safe_divide(df[flux_org_col], df[flux_org_err_col])
    df['nsr_rec'] = safe_divide(df[flux_rec_col], df[flux_rec_err_col])
    df['nsr_noise'] = safe_divide(df[flux_noise_col], df[flux_noise_err_col])
    for index, exp_ratio in enumerate(np.sort(df['exp_ratio'].unique())):
        sub_df = df
        i = (index % 2)
        label = r"Total"
        
        if int(exp_ratio) != 0:
            sub_df = df[df['exp_ratio'] == exp_ratio]
            label = rf'$\gamma={int(exp_ratio)}$'
        
        ax = axes[i, j]
        # Define logarithmic bins
        positive_org_flux = sub_df.loc[positive_mask(sub_df, flux_org_col), flux_org_col]
        if positive_org_flux.empty:
            logging.warning(f"No positive flux values for SNR plot at exp_ratio {exp_ratio}, skipping.")
            continue
        bins = np.logspace(np.log10(positive_org_flux.min()), np.log10(positive_org_flux.max()), 20)

        # Compute binned statistics for each dataset
        gt_x, gt_median, gt_p25, gt_p75 = binned_median(sub_df[flux_org_col], sub_df['nsr_org'], bins)
        noisy_x, noisy_median, noisy_p25, noisy_p75 = binned_median(sub_df[flux_noise_col], sub_df['nsr_noise'], bins)
        rec_x, rec_median, rec_p25, rec_p75 = binned_median(sub_df[flux_rec_col], sub_df['nsr_rec'], bins)

        # Ground Truth
        ax.plot(gt_x, gt_median, color="black", label="Ground Truth")
        ax.fill_between(gt_x, gt_p25, gt_p75, color="black", alpha=0.2)

        # Noisy
        ax.plot(noisy_x, noisy_median, color="darkblue", label="Noisy")
        ax.fill_between(noisy_x, noisy_p25, noisy_p75, color="darkblue", alpha=0.2)

        # Reconstructed
        ax.plot(rec_x, rec_median, color="darkgreen", label=r"Reconstructed")
        ax.fill_between(rec_x, rec_p25, rec_p75, color="darkgreen", alpha=0.2)

        ax.set_xscale("log")
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        ax.set_title(f"{label}")
        ax.legend()

        if j == 0:
            ax.set_ylabel(r'SNR$')
        if i == 1:
            ax.set_xlabel(r'Flux $[erg\ s^{-1}\ cm^{-2}\ \AA^{-1}]$')
        if i == 1:
            j += 1
    # Add a main title
    fig.suptitle(r"SNR vs. Flux", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(output_filepath, dpi=500)
    ###plt.show()
    plt.close()

def main():
    """Load config and produce photometric plots/metrics for one parquet input.

    Input files
    -----------
    ``<uncropped_output_dir>/<photometrical_data_filename>`` : Parquet
        Raw photometric catalogue produced by the evaluation pipeline.
    ``<uncropped_output_dir>/edited_<photometrical_data_filename>`` : Parquet (cached)
        Pre-processed version with flux/magnitude columns added by :func:`add_columns`.
    ``<data_dir>/<metadata_filename>`` : CSV
        Enriched metadata; used by :func:`create_table` to filter eval/test datasets.
    ``<uncropped_output_dir>/<metrics_filename>`` : CSV
        Per-crop detection metrics produced by the evaluation pipeline.

    Output files
    ------------
    ``<output_dir>/<subset>/<subset>_<plot>.png``
        One PNG per plot function per brightness subset (``25`` / ``all``).
    ``<output_dir>/all_metrics.csv``
        Combined detection metrics table for this run.
    """
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    overrides = parse_config_overrides()
    cfg = load_config(**overrides)
    logging.info("Config loaded.")

    vis_cfg = cfg['prepare_plots']

    _output_dir        = vis_cfg['output_dir']

    _parquet_filepath  = os.path.join(
        vis_cfg['uncropped_output_dir'],
        vis_cfg['photometrical_data_filename']
    )

    _metrics_filename  = vis_cfg['uncropped_results_csv']
    _metadata_filepath = vis_cfg['metadata_filepath']
    _rec_cmap          = vis_cfg['rec_cmap']
    _noise_cmap        = vis_cfg['noise_cmap']
    _norm_quantiles    = vis_cfg['norm_quantiles']
    
    if len(_norm_quantiles) != 2:
        raise ValueError('visualization.prepare_plots.norm_quantiles must contain exactly two values.')
    if _norm_quantiles[0] >= _norm_quantiles[1]:
        raise ValueError('visualization.prepare_plots.norm_quantiles must be strictly increasing.')
    if _norm_quantiles[0] < 0 or _norm_quantiles[1] > 100:
        raise ValueError('visualization.prepare_plots.norm_quantiles must stay within [0, 100].')
    path_to_data = _parquet_filepath
    if not os.path.exists(path_to_data):
        raise FileNotFoundError(f"Parquet file not found: {path_to_data}")

    edited_filepath = os.path.join(os.path.dirname(path_to_data), f'edited_{os.path.basename(path_to_data)}')
    if not os.path.exists(edited_filepath):
        df = pd.read_parquet(path_to_data)
        try:
            df = add_columns(df, vis_cfg.get('jansky_factor', None))
        except TypeError:
            df = add_columns(df)
        df.to_parquet(edited_filepath, index=False)
    else:
        df = pd.read_parquet(edited_filepath)

    _metrics_filepath = os.path.join(os.path.dirname(path_to_data), _metrics_filename)
    metrics = create_table(_metadata_filepath, _metrics_filepath)

    bright_df = df[(df['abmag_cflux_rec'] <= 25.0) | (df['abmag_cflux_org'] <= 25.0)].reset_index(drop=True)
    bright_metrics = new_metrics(bright_df, exp_time_col='exp_ratio')
    bright_metrics = bright_metrics.rename(
        columns={col: 'bright_' + col for col in bright_metrics.columns if col != 'exp_ratio'}
    )

    all_metrics = pd.merge(bright_metrics, metrics, on=['exp_ratio']).sort_values(by=['exp_ratio'])

    bright_df = df[df['abmag_cflux_org'] <= 25.0].dropna(subset=['abmag_cflux_org']).reset_index(drop=True)
    for my_df, my_tag in zip([bright_df, df], ('25', 'all')):
        my_output_dir = os.path.join(_output_dir, my_tag)
        columns = {
            'org_flux': 'a_cflux_org',
            'rec_flux': 'a_cflux_rec',
            'noise_flux': 'a_cflux_noise',
            'org_kron': 'kron_radius_org',
            'rec_kron': 'kron_radius_rec',
            'noise_kron': 'kron_radius_noise',
            'org_mag': 'abmag_cflux_org',
            'org_flux_err': 'a_flux_err_org',
            'noise_flux_err': 'a_flux_err_noise',
            'rec_flux_err': 'a_flux_err_rec',
        }
        plot_specs = build_hexbin_plot_specs(_rec_cmap, _noise_cmap)

        snr_filename = vis_cfg.get('snr_filename', 'snr.png')
        for spec in plot_specs:
            try:
                render_hexbin_plot(
                    my_df,
                    os.path.join(my_output_dir, my_tag + "_" + spec['output_filename']),
                    spec,
                    columns,
                    _norm_quantiles,
                )
            except Exception as e:
                logging.warning(f"Error in {spec['output_filename']}: {e}")

        try:
            create_flux_flux_error_diagram(
                my_df,
                columns['org_flux'],
                columns['noise_flux'],
                columns['rec_flux'],
                columns['org_flux_err'],
                columns['noise_flux_err'],
                columns['rec_flux_err'],
                os.path.join(my_output_dir, my_tag + "_" + snr_filename),
            )
        except Exception as e:
            logging.warning(f"Error in create_flux_flux_error_diagram: {e}")

        logging.info(f"Saved plots for subset '{my_tag}' -> {my_output_dir}")

    out_csv = os.path.join(_output_dir, 'all_metrics.csv')
    ensure_parent_dir_exists(out_csv)
    all_metrics.to_csv(out_csv, index=False)
    logging.info(f"All metrics written to {out_csv} ({len(all_metrics)} rows).")


if __name__ == "__main__":
    main()