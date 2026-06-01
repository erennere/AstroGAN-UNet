"""MAST metadata query and download helpers."""

import io
import os, logging
import asyncio, aiohttp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from astroquery.mast.missions import MastMissions
from astroquery.mast import Observations
from astropy.io import fits
from astropy.table import vstack
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections.abc import Sized
from src.training.utils import ensure_parent_dir_exists
from starter import load_config, parse_config_overrides  #sym:parse_config_overrides

def filter_out_mast(mission, filters, query_limit=5000):
    """Retrieve and filter metadata from MAST for a given mission.

    Queries the MAST API with pagination and returns the result as a
    ``pandas.DataFrame``.

    Parameters
    ----------
    mission : str
        Space mission name (e.g. ``'HST'``, ``'TESS'``).
    filters : dict
        Filter criteria; keys are column names, values are filter values.
        Only keys that exist in the mission's column list are forwarded.

    Returns
    -------
    pandas.DataFrame or None
        Filtered metadata, or ``None`` if an error occurred or no data was
        found.
    """

    kwargs = {}
    length = 1
    offset = 0
    limit = int(query_limit)
    results = []

    try:
        logging.info('Querying MAST mission=%s with %d filters.', mission, len(filters) if isinstance(filters, dict) else 0)
        missions = MastMissions(mission=mission)
        # getting the column names in the mission
        if missions:
            columns = missions.get_column_list()
            columns = columns['name']
        else:
            return None

        # checking whether the filter is a dictionary and whether the given filters are covered by the mission
        if isinstance(filters, dict):
            keys = filters.keys()
            for key in keys:
                if key in columns:
                    kwargs[key] = filters[key]

        # iterating with offset as there seems to be a limit of maximum returnees
        index = 0
        while length:
            result = missions.query_criteria(select_cols=[],  # type: ignore[attr-defined]
                                             limit=limit,
                                             offset=offset,
                                             **kwargs)
            if result is not None:
                if isinstance(result, Sized):
                    length = len(result)
                else:
                    length = 0
                if length:
                    results.append(result)
            else:
                length = 0
            logging.debug('MAST page %d fetched %d rows (offset=%d).', index, length, offset)
            offset += length
            index += 1

        if not results:
            logging.warning('MAST query returned no results.')
            return None
        results = vstack(results).to_pandas()
        logging.info('MAST query complete: %d rows collected.', len(results))
        return results
    except Exception as err:
        logging.warning(f"An error occurred while retrieving metadata: {err}")
        return None
    
def plot_histogram(data, bins=20, label=None, xlabel='Exposure Time (s)', ylabel='Frequency', title='Histogram of Exposure', output_filename='histogram.png',loc='upper left', c='b', rotation=60):
    """Plot a histogram of *data* and save the figure to *output_filename*.

    Statistical summary (mean, median, std, variance) is shown in the legend
    when *label* is ``None``.

    Parameters
    ----------
    data : array-like
        Input data to plot.
    bins : int, optional
        Number of histogram bins. Default is 20.
    label : str or None, optional
        Legend label. Auto-generated from statistics when ``None``.
    xlabel : str, optional
        X-axis label. Default is ``'Exposure Time (s)'``.
    ylabel : str, optional
        Y-axis label. Default is ``'Frequency'``.
    title : str, optional
        Plot title. Default is ``'Histogram of Exposure'``.
    output_filename : str, optional
        Output image file path. Default is ``'histogram.png'``.
    loc : str, optional
        Legend location. Default is ``'upper left'``.
    c : str, optional
        Histogram bar colour. Default is ``'b'``.
    rotation : int, optional
        X-axis tick label rotation in degrees. Default is 60.

    Returns
    -------
    None
    """
    # Calculate statistics
    mean = data.mean()
    median = data.median()
    std = data.std()
    var = data.var()

    if label is None:
        label = f'mean: {round(mean, 2)}, median: {round(median, 2)}\nstd: {round(std, 2)}, var: {round(var, 2)}\n N: {len(data)}'

    # Create histogram
    counts, bins_edges, patches = plt.hist(data, bins=bins, density=False, alpha=0.75, color=c, edgecolor='black', label=label)
    xticks = np.linspace(bins_edges.min(), bins_edges.max(), bins+1)
    diff = (xticks[1] - xticks[0])/2
    plt.xticks(np.linspace(bins_edges.min() + diff, bins_edges.max() + diff, bins+1), rotation=rotation)
    plt.grid()
    plt.axvline(mean, color='r', linestyle='--', label='mean')
    plt.axvline(median, color='purple', linestyle='--', label='median')
    plt.legend(loc=loc)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yscale('log')
    plt.title(title)
    plt.tight_layout()
    ensure_parent_dir_exists(output_filename)
    plt.savefig(output_filename, dpi=300)
    plt.show()

async def download_image(id_, url, save_dir, session, semaphore, filename, timeout_seconds=15):
    """Download a single FITS image from *url* and save it to *save_dir*.

    Parameters
    ----------
    id_ : str
        Unique identifier for the image (used for logging).
    url : str
        URL from which to download the image.
    save_dir : str
        Directory where the FITS file is written.
    session : aiohttp.ClientSession
        Active HTTP session.
    semaphore : asyncio.Semaphore
        Concurrency limiter.
    filename : str
        Filename under which the image is saved.

    Returns
    -------
    bool
        ``True`` if the image was downloaded and saved successfully,
        ``False`` otherwise.
    """
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    if os.path.exists(filepath):
        logging.debug('File already exists, skipping download: %s', filepath)
        return True
    
    async with semaphore:
        try:
            if session is None:
                session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout_seconds))
            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.read()
                    with fits.open(io.BytesIO(content)) as hdul:
                        hdul.writeto(filepath, overwrite=True)
                    logging.debug('Downloaded %s -> %s', id_, filepath)
                    return True 
                logging.debug('Download skipped for %s due to status=%s', id_, response.status)
        except Exception as err:
            logging.warning(f'an error occured while downloading: {err}')
        return False

async def download_images(ids, urls, save_dir, max_requests=5, reset_after=10, timeout_seconds=15):
    """Download multiple FITS images from *urls* asynchronously.

    Parameters
    ----------
    ids : iterable
        Unique identifiers for each image.
    urls : iterable
        Download URLs, one per identifier.
    save_dir : str
        Directory where downloaded images are saved.
    max_requests : int, optional
        Maximum number of concurrent downloads. Default is 5.
    reset_after : int, optional
        Number of requests after which the HTTP session is recycled.
        Default is 10.

    Returns
    -------
    None
    """
    logging.info('Starting bulk download to %s with max_requests=%d reset_after=%d', save_dir, max_requests, reset_after)
    semaphore = asyncio.Semaphore(max_requests)
    session = None
    success_count = 0
    attempted_count = 0
    try:
        for i, (id_, url) in enumerate(zip(ids, urls)):
            if i % reset_after == 0:
                if session:
                    await session.close()
                session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout_seconds))
            if pd.isna(url):
                continue
            attempted_count += 1
            if url is None or pd.isna(url) or str(url).strip() == '':
                continue
            filename = os.path.basename(url)
            downloaded = await download_image(id_, url, save_dir, session, semaphore, filename, timeout_seconds)
            if downloaded:
                success_count += 1
    except Exception as err:
        logging.warning(f'an error occurred while downloading urls: {err}')
    if session:
        await session.close()
    logging.info('Bulk download finished: %d/%d files downloaded.', success_count, attempted_count)
    return True

def _chunked(items, chunk_size):
    """Yield fixed-size chunks from *items*."""
    if chunk_size <= 0:
        chunk_size = 1
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]

def _choose_best_product(products_df, prefer_token='drz'):
    """Select the best FITS product row from an Observations product table."""
    if products_df is None or products_df.empty:
        return None

    df = products_df.copy()
    filename_col = 'productFilename' if 'productFilename' in df.columns else None
    
    if filename_col is not None:
        filename = df[filename_col].fillna('').astype(str)
        df = df[filename.str.lower().str.endswith(f'{prefer_token}.fits')]
        if df.empty:
            return None

        preferred = df[df[filename_col].str.contains(prefer_token, case=False, na=False)]
        if not preferred.empty:
            df = preferred

    if 'productType' in df.columns:
        science = df[df['productType'].astype(str).str.upper() == 'SCIENCE']
        if not science.empty:
            df = science

    if 'calib_level' in df.columns:
        calib_numeric = pd.to_numeric(df['calib_level'], errors='coerce')
        df = df.assign(_calib_level_num=calib_numeric).sort_values('_calib_level_num', ascending=False, na_position='last')

    return df.iloc[0] if not df.empty else None

def _resolve_products_bulk(dataset_ids, prefer_token='drz', max_retries=2, retry_delay=2.0):
    """Resolve many dataset ids to best product URLs using one Observations query.

    Parameters
    ----------
    dataset_ids : list[str]
        Dataset identifiers to resolve.
    prefer_token : str, optional
        Filename token preference for product selection. Default is ``'drz'``.
    max_retries : int, optional
        Number of retries after the initial attempt for bulk API calls.
        Default is 2.
    retry_delay : float, optional
        Base delay in seconds between retries. Default is 2.0.

    Returns
    -------
    dict
        Mapping ``dataset_id -> resolved_url``.
    """
    if not dataset_ids:
        return {}

    wanted_ids = {str(x) for x in dataset_ids if isinstance(x, str) and x.strip() != ''}
    if not wanted_ids:
        return {}

    attempts = max(1, int(max_retries) + 1)
    for attempt in range(1, attempts + 1):
        try:
            obs = Observations.query_criteria(obs_id=list(wanted_ids))
            if obs is None or len(obs) == 0:
                return {}

            products = Observations.get_product_list(obs)
            if products is None:
                return {}

            products_df = products.to_pandas() if hasattr(products, 'to_pandas') else pd.DataFrame(products)
            if products_df.empty:
                return {}
            key_col = next((col for col in ['obs_id', 'obsid', 'obsID'] if col in products_df.columns), None)
            if key_col is None:
                logging.warning('Products table is missing obs-id key column; cannot map products to dataset ids.')
                return {}

            products_df[key_col] = products_df[key_col].str.upper().astype(str)
            products_df = products_df[products_df[key_col].isin(wanted_ids)]
            if products_df.empty:
                return {}

            resolved = {}
            for dataset_id, group in products_df.groupby(key_col, sort=False):
                best = _choose_best_product(group, prefer_token=prefer_token)
                if best is None:
                    continue

                data_uri = best.get('dataURI', None)
                url = None
                if isinstance(data_uri, str) and data_uri.strip() != '':
                    url = f'https://mast.stsci.edu/api/v0.1/Download/file?uri={data_uri}'
                if url is None and 'dataURL' in best:
                    data_url = best.get('dataURL', None)
                    if isinstance(data_url, str) and data_url.strip() != '':
                        url = data_url

                if url:
                    resolved[str(dataset_id)] = url

            return resolved
        except Exception as err:
            if attempt < attempts:
                delay = retry_delay * attempt
                logging.warning(
                    'Bulk resolution attempt %d/%d failed for %d ids: %s. Retrying in %.1fs.',
                    attempt,
                    attempts,
                    len(wanted_ids),
                    err,
                    delay,
                )
                time.sleep(delay)
            else:
                logging.warning(
                    'Failed bulk product resolution after %d attempts for %d dataset ids: %s',
                    attempts,
                    len(wanted_ids),
                    err,
                )
                return {}

def merge_products_with_metadata(
    table,
    id_column,
    url_column,
    max_workers=8,
    prefer_token='drz',
    chunk_size=500,
    min_chunk_size=1,
    max_retries=2,
    retry_delay=2.0,
):
    """Merge Observations product URLs into metadata rows by dataset id.

    Parameters
    ----------
    table : pandas.DataFrame
        Metadata from ``filter_out_mast``.
    id_column : str
        Dataset-id column (e.g. ``sci_data_set_name``).
    url_column : str
        Output URL column to populate.
    max_workers : int, optional
        Process workers for chunk resolution. Default is 8.
    prefer_token : str, optional
        Filename token preference for product selection. Default is ``'drz'``.
    chunk_size : int, optional
        Number of unresolved ids per bulk Observations query chunk.
    max_retries : int, optional
        Number of retries for a failed bulk Observations resolution call.
    retry_delay : float, optional
        Base retry delay in seconds. Delay is scaled per attempt.

    Returns
    -------
    pandas.DataFrame
        Metadata table enriched with resolved product URLs.
    """
    if not isinstance(table, pd.DataFrame) or table.empty:
        return table
    if id_column not in table.columns:
        logging.warning('Cannot merge products: missing id column %s', id_column)
        return table

    merged = table.copy()
    if url_column not in merged.columns:
        merged[url_column] = None

    unresolved_ids = (
        merged.loc[merged[url_column].isna(), id_column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    if not unresolved_ids:
        return merged

    logging.info('Resolving product URLs for %d dataset ids via bulk Observations.get_product_list.', len(unresolved_ids))
    resolved_map = {}
    chunk_size = max(int(min_chunk_size), int(chunk_size))
    chunks = list(_chunked(unresolved_ids, chunk_size))
    total_chunks = len(chunks)

    if total_chunks == 1 or int(max_workers) <= 1:
        for idx, id_chunk in enumerate(chunks, start=1):
            logging.info('Resolving chunk %d/%d (%d ids).', idx, total_chunks, len(id_chunk))
            resolved_map.update(
                _resolve_products_bulk(
                    id_chunk,
                    prefer_token=prefer_token,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                )
            )
    else:
        worker_count = min(int(max_workers), total_chunks)
        logging.info('Resolving chunks with ProcessPoolExecutor (workers=%d).', worker_count)
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    _resolve_products_bulk,
                    id_chunk,
                    prefer_token,
                    max_retries,
                    retry_delay,
                ): (idx, id_chunk)
                for idx, id_chunk in enumerate(chunks, start=1)
            }
            for future in as_completed(futures):
                idx, id_chunk = futures[future]
                chunk_len = len(id_chunk)
                logging.info('Resolved chunk %d/%d (%d ids).', idx, total_chunks, chunk_len)
                try:
                    resolved_map.update(future.result())
                except Exception as err:
                    logging.warning('Chunk %d/%d failed in process pool: %s. Retrying in parent process.', idx, total_chunks, err)
                    resolved_map.update(
                        _resolve_products_bulk(
                            id_chunk,
                            prefer_token=prefer_token,
                            max_retries=max_retries,
                            retry_delay=retry_delay,
                        )
                    )

    if resolved_map:
        mapped = merged[id_column].astype(str).map(resolved_map)
        merged[url_column] = merged[url_column].where(merged[url_column].notna(), mapped)

    resolved_count = merged[url_column].notna().sum()
    logging.info('Product merge complete: %d/%d rows have %s.', resolved_count, len(merged), url_column)
    return merged

def main():
    """Run MAST metadata/query URL-resolution steps from config.yaml (no fallback defaults)."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    overrides = parse_config_overrides()  # parses sys.argv by default
    root_cfg = load_config(**overrides)
    if not isinstance(root_cfg, dict):
        raise TypeError('Loaded config must be a mapping.')
    if 'mast' not in root_cfg:
        raise KeyError("Missing required top-level config section: 'mast'")
    if not isinstance(root_cfg['mast'], dict):
        raise TypeError("config['mast'] must be a mapping.")

    cfg = dict(root_cfg['mast'])
    required_keys = [
        'mission',
        'filters',
        'main_column',
        'max_requests',
        'reset_after',
        'max_workers',
        'download',
        'chunk_size',
        'resolve_max_retries',
        'resolve_retry_delay',
        'metadata_output',
        'id_column',
        'url_column',
        'fetch_metadata',
        'resolve_urls',
        'query_limit',
        'http_timeout_seconds',
        'min_chunk_size',
        'prefer_token',
        'save_dir',
    ]
    missing_keys = [k for k in required_keys if k not in cfg]
    if missing_keys:
        logging.error('Missing required mast config keys: %s', ', '.join(missing_keys))
        return

    metadata_output = os.path.abspath(cfg['metadata_output'])
    logging.info('Loaded mast config: mission=%s fetch_metadata=%s resolve_urls=%s',
                 cfg['mission'], cfg['fetch_metadata'], cfg['resolve_urls'])

    table = None
    if cfg['fetch_metadata']:
        table = filter_out_mast(cfg['mission'], cfg['filters'], cfg['query_limit'])
        if table is None or table.empty:
            logging.warning('No metadata returned from MAST. Nothing to write.')
        else:
            table.sort_values(by=[cfg['main_column']], ascending=False, inplace=True)
            ensure_parent_dir_exists(metadata_output)
            table.to_csv(metadata_output, index=False)
            logging.info('Saved metadata table with %d rows to %s', len(table), metadata_output)

    if cfg['resolve_urls']:
        if table is None:
            if os.path.exists(metadata_output):
                logging.info('Loading metadata table from %s for product merge.', metadata_output)
                table = pd.read_csv(metadata_output)
            else:
                logging.warning('Metadata CSV not found: %s', metadata_output)
                return


        table = merge_products_with_metadata(
            table,
            id_column=cfg['id_column'],
            url_column=cfg['url_column'],
            max_workers=cfg['max_workers'],
            prefer_token=cfg['prefer_token'],
            chunk_size=cfg['chunk_size'],
            min_chunk_size=cfg['min_chunk_size'],
            max_retries=cfg['resolve_max_retries'],
            retry_delay=cfg['resolve_retry_delay'],
        )

        ensure_parent_dir_exists(metadata_output)
        table.to_csv(metadata_output, index=False)
        logging.info('Saved merged metadata with product URLs to %s', metadata_output)

        if cfg['download'] and cfg['url_column'] in table.columns:
            valid = table[[cfg['id_column'], cfg['url_column']]].dropna(subset=[cfg['url_column']])
            if not valid.empty:
                asyncio.run(
                    download_images(
                        valid[cfg['id_column']].tolist(),
                        valid[cfg['url_column']].tolist(),
                        cfg['save_dir'],
                        max_requests=cfg['max_requests'],
                        reset_after=cfg['reset_after'],
                        timeout_seconds=cfg['http_timeout_seconds'],
                    )
                )


if __name__ == '__main__':
    main()











