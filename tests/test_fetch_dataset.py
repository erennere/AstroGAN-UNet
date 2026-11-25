import asyncio
import threading

import numpy as np
import pandas as pd
import pytest

from astrounets.data.dataset import build_dataset, load_crops, select_images
from astrounets.data.fits_io import CropStore, read_image, write_image
from astrounets.config import replace as replace_cfg
from astrounets.data.mast import apply_buffer, assign_splits, assign_tiles, download
from conftest import smoke_config


def test_tiles_group_nearby_images():
    ra = np.array([10.1, 10.11, 10.12, 50.1, 50.11, 120.1])
    dec = np.array([0.1, 0.1, 0.11, -5.1, -5.1, 30.1])
    tiles = assign_tiles(ra, dec, tile_arcmin=20.0)
    assert tiles[0] == tiles[1] == tiles[2] and tiles[3] == tiles[4]
    assert len(set(tiles)) == 3
    assert assign_tiles([359.99], [0.1], 20.0)[0] != assign_tiles([0.01], [0.1], 20.0)[0]


def test_splits_keep_tiles_whole_and_fill_every_split():
    rng = np.random.default_rng(0)
    tiles = rng.integers(0, 40, 2000)
    split = assign_splits(tiles, {"train": 60, "val": 20, "test": 20}, seed=1)
    table = pd.DataFrame({"tile": tiles, "split": split})
    assert (table.groupby("tile")["split"].nunique() == 1).all()
    share = table["split"].value_counts(normalize=True)
    assert set(share.index) == {"train", "val", "test"}
    assert share["train"] == pytest.approx(0.6, abs=0.05) and share["test"] == pytest.approx(0.2, abs=0.05)
    assert np.array_equal(split, assign_splits(tiles, {"train": 60, "val": 20, "test": 20}, seed=1))


def test_splits_need_three_tiles():
    with pytest.raises(ValueError, match="at least 3"):
        assign_splits(np.array([0, 0, 1]), {"train": 60, "val": 20, "test": 20}, seed=0)


def test_buffer_removes_images_that_overlap_an_earlier_split():
    ra = np.array([10.0, 10.03, 10.0, 40.0, 40.02])      # 0.03 deg = 1.8' apart
    dec = np.array([0.0, 0.0, 0.04, 0.0, 0.0])
    split = np.array(["train", "val", "test", "val", "test"], dtype=object)
    out = apply_buffer(ra, dec, split, overlap_arcmin=3.1)
    assert out.tolist() == ["train", "buffer", "buffer", "val", "buffer"]
    assert apply_buffer(ra, dec, split, 0.0).tolist() == split.tolist()


def test_download_subset_is_spread_over_splits_and_tiles(cfg):
    from astrounets.data.mast import choose_downloads

    n = 300
    table = pd.DataFrame({"id": [f"I{i}" for i in range(n)], "exposure": 1000.0, "aperture": "IR", "pi": "X",
                          "tile": np.arange(n) % 30, "split": np.array(["train", "val", "test"])[np.arange(n) % 30 % 3]})
    limited = replace_cfg(cfg, {"mast": {"download_limit": 20}, "dataset": {"apertures": [], "exposure_min": 100.0}})
    chosen = choose_downloads(table, limited)
    assert chosen["split"].value_counts().to_dict() == {"train": 12, "val": 4, "test": 4}
    assert chosen.groupby("split")["tile"].nunique().to_dict() == {"test": 4, "train": 10, "val": 4}
    assert len(choose_downloads(table, cfg)) == n


def test_dataset_uses_every_selected_image(mock_data):
    meta = pd.read_parquet(mock_data.mast_dir / "metadata.parquet")
    crops = load_crops(mock_data)
    assert crops["image_id"].nunique() == len(meta) == mock_data.mast.mock_images  # no hidden row cap
    assert set(crops["split"]) == {"train", "val", "test"}
    # a tile is never in two splits, and each crop keeps its image's split
    assert (crops.groupby("tile")["split"].nunique() == 1).all()
    merged = crops.merge(meta[["id", "split"]], left_on="image_id", right_on="id", suffixes=("", "_image"))
    assert (merged["split"] == merged["split_image"]).all()
    assert (crops["valid_fraction"] >= mock_data.dataset.min_valid_fraction).all()
    assert crops["sigma"].between(0.005, 0.05).all()


def test_dataset_split_is_shared_across_datasets(mock_data):
    faber = smoke_config(mock_data.data_dir.parent, datasets={"faber-only": {"pi_names": ["FABER"]}},
                         dataset={"name": "faber-only", "workers": 1})
    build_dataset(faber)
    full, sub = load_crops(mock_data), load_crops(faber)
    assert set(sub["pi"]) == {"FABER"}
    lookup = full.drop_duplicates("image_id").set_index("image_id")["split"]
    assert (sub["split"] == sub["image_id"].map(lookup)).all()


def test_images_with_wrong_filter_are_skipped(mock_data, tmp_path):
    from astrounets.data.dataset import image_crops

    path = tmp_path / "other.fits"
    write_image(path, np.ones((128, 128)), "SCI", {"EXPTIME": 900.0, "PHOTFLAM": 1e-20, "PHOTPLAM": 1.4e4, "FILTER": "F140W"})
    rows, reason = image_crops({"abspath": str(path), "path": "x", "id": "X", "split": "train", "tile": 0, "pi": "A"},
                               mock_data.dataset)
    assert rows == [] and reason == "filter F140W"


def test_selection_filters(mock_data):
    meta = pd.read_parquet(mock_data.mast_dir / "metadata.parquet")
    d = smoke_config(mock_data.data_dir.parent, datasets={"mock": {"pi_names": ["momcheva"]}},
                     dataset={"exposure_min": 1000.0}).dataset
    chosen = select_images(meta, d, mock_data.mast_dir, seed=0)
    assert set(chosen["pi"]) == {"MOMCHEVA"} and (chosen["exposure"] >= 1000).all()


def test_crop_store_reads_the_right_window(mock_data):
    crops = load_crops(mock_data)
    row = crops.iloc[5]
    store = CropStore(mock_data.mast_dir, "SCI", max_open=2)
    window = store.crop(row["path"], row["y"], row["x"], 64)
    whole = read_image(mock_data.mast_dir / row["path"], "SCI")
    expected = np.nan_to_num(whole[row["y"]:row["y"] + 64, row["x"]:row["x"] + 64], nan=0.0)
    assert np.array_equal(window, expected)
    for _, r in crops.head(10).iterrows():  # more files than max_open
        store.crop(r["path"], r["y"], r["x"], 64)
    store.close()


def test_parallel_download_from_local_server(tmp_path):
    from aiohttp import web

    payload = b"SIMPLE  =                    T" + b" " * 2850
    started, port_box = threading.Event(), {}

    def serve():
        loop = asyncio.new_event_loop()
        app = web.Application()
        app.router.add_get("/{name}", lambda request: web.Response(body=payload))
        runner = web.AppRunner(app)
        loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, "127.0.0.1", 0)
        loop.run_until_complete(site.start())
        port_box["port"] = site._server.sockets[0].getsockname()[1]
        port_box["loop"] = loop
        started.set()
        loop.run_forever()

    threading.Thread(target=serve, daemon=True).start()
    started.wait(5)
    base = f"http://127.0.0.1:{port_box['port']}"
    urls = {f"ID{i}": f"{base}/id{i}_drz.fits" for i in range(6)}
    present = download(urls, tmp_path / "originals", workers=3, timeout=10)
    assert set(present) == set(urls) and all(p.read_bytes() == payload for p in present.values())
    (tmp_path / "originals" / "id0_drz.fits").write_bytes(b"keep")
    download(urls, tmp_path / "originals", workers=3, timeout=10)
    assert (tmp_path / "originals" / "id0_drz.fits").read_bytes() == b"keep"  # existing files are not re-downloaded
    port_box["loop"].call_soon_threadsafe(port_box["loop"].stop)


def test_fetch_with_mocked_mast(tmp_path, monkeypatch):
    from astrounets.data import mast

    cfg = smoke_config(tmp_path, mast={"mock": False, "download_limit": 5}, dataset={"exposure_min": 100.0})
    ra = [10.1, 10.11, 60.1, 60.11, 120.1, 120.11, 200.1, 200.11, 300.1]
    table = pd.DataFrame({"id": [f"I{i}" for i in range(9)], "exposure": [800.0] * 9, "ra": ra, "dec": [0.1] * 9,
                          "aperture": ["IR"] * 9, "pi": ["A"] * 9, "target": ["T"] * 9})
    calls = {"query": 0, "resolved": [], "downloaded": []}

    def fake_download(urls, folder, workers, timeout):
        calls["downloaded"] += list(urls)
        folder.mkdir(parents=True, exist_ok=True)
        return {i: folder / f"{i.lower()}_drz.fits" for i in urls}

    monkeypatch.setattr(mast, "query_metadata", lambda c: calls.__setitem__("query", calls["query"] + 1) or table.copy())
    monkeypatch.setattr(mast, "resolve_urls", lambda ids, c: calls["resolved"].extend(ids) or {i: f"https://x/{i.lower()}_drz.fits" for i in ids})
    monkeypatch.setattr(mast, "download", fake_download)
    meta = pd.read_parquet(mast.fetch(cfg))
    assert len(meta) == 9 and meta["tile"].nunique() == 5
    assert len(calls["resolved"]) == 5 and sorted(calls["resolved"]) == sorted(calls["downloaded"])  # only the subset
    assert set(meta.loc[meta["path"].notna(), "split"]) == {"train", "val", "test"}
    mast.fetch(cfg)
    assert calls["query"] == 1, "an existing metadata table must be reused unless --refresh"
    assert len(calls["resolved"]) == 5, "URLs already known are not resolved again"


def test_resolve_urls_prefers_the_association_product(cfg, monkeypatch):
    import astroquery.mast as aqm
    from astrounets.data import mast

    products = pd.DataFrame({
        "obs_id": ["ibeu01020", "ibeu01020", "ibeu01020", "ibeu02030"],
        "productFilename": ["ibeu01abq_drz.fits", "ibeu01020_drz.fits", "ibeu01020_flt.fits", "ibeu02030_drz.fits"],
        "dataURI": ["mast:HST/product/ibeu01abq_drz.fits", "mast:HST/product/ibeu01020_drz.fits",
                    "mast:HST/product/ibeu01020_flt.fits", "mast:HST/product/ibeu02030_drz.fits"],
    })

    class FakeTable:
        def to_pandas(self):
            return products.copy()

    monkeypatch.setattr(aqm.Observations, "query_criteria", lambda **kw: "obs")
    monkeypatch.setattr(aqm.Observations, "get_product_list", lambda obs: FakeTable())
    urls = mast.resolve_urls(["IBEU01020", "IBEU02030", "IBEU09999"], cfg)
    assert urls == {"IBEU01020": "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:HST/product/ibeu01020_drz.fits",
                    "IBEU02030": "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:HST/product/ibeu02030_drz.fits"}


def test_crop_table_must_match_the_config(mock_data):
    from astrounets.config import replace

    assert len(load_crops(mock_data)) > 0
    other = replace(mock_data, {"dataset": {"min_valid_fraction": 0.5}})
    with pytest.raises(RuntimeError, match="other dataset settings"):
        load_crops(other)
    with pytest.raises(FileNotFoundError, match="dataset.name=mock-faber"):
        load_crops(mock_data, "mock-faber")


def test_node_scratch_copies_images_inside_slurm_jobs(mock_data, tmp_path, monkeypatch):
    from astrounets.config import replace
    from astrounets.data.scratch import image_root

    files = sorted(load_crops(mock_data)["path"].unique())[:3]
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    assert image_root(mock_data, files) == mock_data.mast_dir, "nothing is copied outside SLURM jobs"
    monkeypatch.setenv("SLURM_JOB_ID", "42")
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    root = image_root(mock_data, files)
    assert root == tmp_path / "astrounets"
    for rel in files:
        assert (root / rel).read_bytes() == (mock_data.mast_dir / rel).read_bytes()
    assert image_root(mock_data, files) == root  # second call finds the copies
    off = replace(mock_data, {"paths": {"node_scratch": False}})
    assert image_root(off, files) == mock_data.mast_dir
    store = CropStore(root, "SCI")
    row = load_crops(mock_data).query("path == @files[0]").iloc[0]
    assert np.array_equal(store.crop(row["path"], row["y"], row["x"], 64),
                          CropStore(mock_data.mast_dir, "SCI").crop(row["path"], row["y"], row["x"], 64))


def test_atomic_write_leaves_nothing_on_failure(tmp_path):
    from astrounets.files import atomic, write_text

    target = tmp_path / "sub" / "out.txt"
    write_text(target, "first")
    with pytest.raises(ZeroDivisionError):
        with atomic(target) as tmp:
            tmp.write_text("half")
            1 / 0
    assert target.read_text() == "first" and sorted(p.name for p in target.parent.iterdir()) == ["out.txt"]
