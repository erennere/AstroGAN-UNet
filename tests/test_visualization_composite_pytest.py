"""Tests for composite visualization functions in src/visualization/prepare_images.py."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.visualization.prepare_images import (
    create_composite_plot,
    create_composite_plot_detections,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _rng():
    return np.random.default_rng(123)


def _image(rng):
    return rng.uniform(0, 100, (64, 64)).astype(np.float32)


def _ellipse_patch():
    from matplotlib.patches import Ellipse
    return Ellipse((0, 0), 4, 2)


# ---------------------------------------------------------------------------
# create_composite_plot
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_create_composite_plot_saves_png_to_disk(tmp_path: Path):
    import matplotlib.pyplot as plt
    rng = _rng()
    org = _image(rng)
    noisy = [_image(rng), _image(rng)]
    recs = [_image(rng), _image(rng)]
    gammas = [2.0, 4.0]
    output_path = str(tmp_path / 'composite.png')

    create_composite_plot(org, noisy, recs, gammas, label='Test', output_filepath=output_path)
    plt.close('all')

    assert Path(output_path).exists()
    assert Path(output_path).stat().st_size > 0


@pytest.mark.unit
def test_create_composite_plot_single_gamma(tmp_path: Path):
    import matplotlib.pyplot as plt
    rng = _rng()
    org = _image(rng)
    output_path = str(tmp_path / 'single_gamma.png')

    create_composite_plot(org, [_image(rng)], [_image(rng)], [8.0], label='Single', output_filepath=output_path)
    plt.close('all')

    assert Path(output_path).exists()


# ---------------------------------------------------------------------------
# create_composite_plot_detections
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_create_composite_plot_detections_saves_png_to_disk(tmp_path: Path):
    import matplotlib.pyplot as plt
    rng = _rng()
    org = _image(rng)
    noisy = [_image(rng), _image(rng)]
    recs = [_image(rng), _image(rng)]
    gammas = [2.0, 4.0]

    # Each ellipses element: [org_ellipses, rec_ellipses, noisy_ellipses]
    ellipses = [
        [[_ellipse_patch()], [_ellipse_patch()], [_ellipse_patch()]],
        [[_ellipse_patch()], [_ellipse_patch()], [_ellipse_patch()]],
    ]

    output_path = str(tmp_path / 'detections.png')

    create_composite_plot_detections(
        org, noisy, recs, gammas, ellipses, label='Detection Test', output_filepath=output_path
    )
    plt.close('all')

    assert Path(output_path).exists()
    assert Path(output_path).stat().st_size > 0
