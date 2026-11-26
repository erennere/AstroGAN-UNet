import asyncio
import json
import threading

import numpy as np
import pandas as pd
import pytest

from astrounets.data.dataset import build_dataset, load_crops, select_images
from astrounets.data.fits_io import CropStore, read_image, write_image
from astrounets.config import replace as replace_cfg
from astrounets.data.mast import apply_buffer, assign_splits, assign_splits_per_program, assign_tiles, download
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


def test_big_programs_are_in_every_split_and_tiles_stay_whole():
    """A program observing a few patches of sky (here 10 tiles) is shared out over train, val and test
    instead of landing wherever the global draw puts its tiles; small programs are split together."""
    rng = np.random.default_rng(3)
    big = pd.DataFrame({"tile": np.repeat(np.arange(10), 30), "pi": "DALCANTON"})        # 10 tiles, 300 images
    small = pd.DataFrame({"tile": 100 + np.arange(60), "pi": [f"P{i}" for i in range(60)]})  # 60 one-tile programs
    mixed = pd.DataFrame({"tile": [0, 0, 5], "pi": ["OTHER", "OTHER", "OTHER"]})          # minor guests in big tiles
    table = pd.concat([big, small, mixed], ignore_index=True).sample(frac=1.0, random_state=1)
    shares = {"train": 60, "val": 20, "test": 20}
    split = assign_splits_per_program(table["tile"].to_numpy(), table["pi"].to_numpy(), shares, seed=4)
    table["split"] = split
    assert (table.groupby("tile")["split"].nunique() == 1).all()
    assert set(table.loc[table["pi"] == "DALCANTON", "split"]) == {"train", "val", "test"}
    assert set(table.loc[table["pi"].str.startswith("P"), "split"]) == {"train", "val", "test"}
    again = assign_splits_per_program(table["tile"].to_numpy(), table["pi"].to_numpy(), shares, seed=4)
    assert np.array_equal(split, again)
    big_share = table.loc[table["pi"] == "DALCANTON", "split"].value_counts(normalize=True)
    assert big_share["train"] == pytest.approx(0.6, abs=0.11)


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
    unlimited = replace_cfg(cfg, {"dataset": {"apertures": [], "exposure_min": 100.0}})
    assert len(choose_downloads(table, unlimited)) == n


def test_full_download_skips_images_no_dataset_uses(cfg):
    from astrounets.data.mast import choose_downloads

    table = pd.DataFrame({"id": [f"I{i}" for i in range(6)], "tile": 0,
                          "exposure": [1000.0, 50.0, 1000.0, 1000.0, 1000.0, 1000.0],
                          "aperture": ["IR", "IR", "IRSUB64", "IR", "IR", "IR"],
                          "pi": ["A", "A", "A", "FABER", "B", "A"],
                          "split": ["train", "train", "train", "val", "test", "buffer"]})
    full = replace_cfg(cfg, {"dataset": {"apertures": ["IR"], "exposure_min": 100.0}})
    assert choose_downloads(table, full)["id"].tolist() == ["I0", "I3", "I4"]  # short, wrong aperture, buffer: no
    only_faber = replace_cfg(full, {"datasets": {name: {"pi_names": ["FABER"]} for name in full.datasets}})
    assert choose_downloads(table, only_faber)["id"].tolist() == ["I3"]  # no dataset wants the other PIs


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
    assert crops["sigma"].between(0.005, 0.1).all()


def test_noise_is_measured_on_empty_sky_and_modelled_in_crowded_fields(mock_data):
    """Sparse mock images: the noise measured on their empty sky is the injected one. Crowded images
    (thousands of faint stars): the spread of their pixels is mostly stars; their noise comes from the
    photon + read noise expectation, scaled by k measured on the sparse crops, and they are marked crowded."""
    from astrounets.data.mock import MOCK_K, SKY, UNRESOLVED, is_crowded, noise_sigma

    crops = load_crops(mock_data)
    meta = pd.read_parquet(mock_data.mast_dir / "metadata.parquet").set_index("id")
    crowded_image = np.array([is_crowded(int(i[4:])) for i in crops["image_id"]])
    truth = np.array([noise_sigma(meta.loc[i, "exposure"], SKY + UNRESOLVED * c) for i, c in zip(crops["image_id"], crowded_image)])
    error = crops["sigma"].to_numpy() / truth - 1
    summary = json.loads((mock_data.dataset_dir() / "summary.json").read_text())
    assert summary["noise_scale_k"] == pytest.approx(MOCK_K, rel=0.03)
    assert np.abs(error[~crowded_image]).max() < 0.1 and crops.loc[~crowded_image, "noise_measured"].all()
    assert not crops.loc[~crowded_image, "crowded"].any()
    assert crops.loc[crowded_image, "crowded"].mean() > 0.8
    from_model = crowded_image & ~crops["noise_measured"].to_numpy()
    assert from_model.sum() > 0.5 * crowded_image.sum() and np.abs(error[from_model]).max() < 0.1
    assert np.abs(error[crowded_image]).max() < 0.35  # measured ones pass the excess check, so are close
    assert set(crops.loc[crowded_image, "split"]) == {"train", "val", "test"}
    # the clipped spread of all pixels (the previous estimate) is far off in a crowded image
    from astrounets.data.dataset import measure_background

    first = crops[crowded_image].iloc[0]
    image = read_image(mock_data.mast_dir / first["path"], mock_data.dataset.hdu)
    _, rms = measure_background(image, mock_data.dataset)
    old = np.median(rms[np.isfinite(image) & (image != 0)])
    assert old / noise_sigma(meta.loc[first["image_id"], "exposure"], SKY + UNRESOLVED) > 1.5


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


def test_atomic_write_leaves_nothing_on_failure(tmp_path):
    from astrounets.files import atomic, write_text

    target = tmp_path / "sub" / "out.txt"
    write_text(target, "first")
    with pytest.raises(ZeroDivisionError):
        with atomic(target) as tmp:
            tmp.write_text("half")
            1 / 0
    assert target.read_text() == "first" and sorted(p.name for p in target.parent.iterdir()) == ["out.txt"]


def test_low_weight_pixels_are_no_data(mock_data):
    """The mock "death star" (drizzle weight ~0, junk pixel values) must never reach training or evaluation."""
    from astrounets.data.fits_io import read_weight
    from astrounets.data.pairs import noisy_pair

    crops = load_crops(mock_data)
    assert (crops["weight_cut"] > 0).all()
    image = crops[crops["image_id"] == "MOCK0000"]  # even-numbered mock images have a dead patch
    weight = read_weight(mock_data.mast_dir / image["path"].iloc[0], "WHT")
    cut = float(image["weight_cut"].iloc[0])
    store = CropStore(mock_data.mast_dir, "SCI", "WHT")
    hit = 0
    for _, row in image.iterrows():
        low = weight[row["y"]:row["y"] + 64, row["x"]:row["x"] + 64] < cut
        masked = store.crop(row["path"], row["y"], row["x"], 64, cut)
        raw = store.crop(row["path"], row["y"], row["x"], 64)
        assert (masked[low] == 0).all() and np.array_equal(masked[~low], raw[~low])
        if low.any() and (raw[low] != 0).any():
            hit += 1
            clean, noisy = noisy_pair(store, row.to_dict(), 8.0, 0, 64)
            assert (noisy[low] == 0).all()
            assert row["valid_fraction"] <= 1 - low.mean() + 1e-9
    assert hit >= 1, "the dead patch lies in at least one usable crop"


def test_weight_cut_can_be_switched_off(mock_data, tmp_path):
    from astrounets.data.dataset import image_crops

    meta = pd.read_parquet(mock_data.mast_dir / "metadata.parquet")
    row = {**meta.iloc[0].to_dict(), "abspath": str(mock_data.mast_dir / meta.iloc[0]["path"])}
    on, _ = image_crops(row, mock_data.dataset)
    off, _ = image_crops(row, replace_cfg(mock_data, {"dataset": {"min_weight_fraction": 0.0}}).dataset)
    assert all(c["weight_cut"] > 0 for c in on) and all(c["weight_cut"] == 0 for c in off)
    assert sum(c["valid_fraction"] for c in on) < sum(c["valid_fraction"] for c in off)
