"""Unit tests for changepoints.py.

These tests can be run from the root directory using:
    pytest src/clehrity/test_changepoints.py
"""

import anndata as ad  # type: ignore
import numpy as np
import pandas as pd
import pytest

from clehrity.changepoints import discontinuities
from clehrity.changepoints import non_monotonicities


# Sample data for testing
N_SAMPLES = 100
N_FEATURES = 3
SAMPLE_OBS = {
    f"feature{i}": np.random.uniform(-1, 1, size=N_SAMPLES) for i in range(N_FEATURES)
}
SAMPLE_OBS["outcome"] = np.logical_or(
    SAMPLE_OBS["feature1"] > 0, SAMPLE_OBS["feature2"] > 0
)
SAMPLE_DF = pd.DataFrame(SAMPLE_OBS, dtype=np.float32)
SAMPLE_ANNDATA = ad.AnnData(SAMPLE_DF)


@pytest.fixture
def sample_anndata() -> ad.AnnData:
    """Fixture to provide a sample AnnData object."""
    return SAMPLE_ANNDATA.copy()


def test_non_monotonicities_returns_dataframe(sample_anndata: ad.AnnData) -> None:
    """Test non_monotonicities function returns a DataFrame."""
    result = non_monotonicities(sample_anndata, "outcome")
    assert isinstance(result, pd.DataFrame)


def test_discontinuities_returns_dataframe(sample_anndata: ad.AnnData) -> None:
    """Test discontinuities function returns a DataFrame."""
    result = discontinuities(sample_anndata, "outcome", min_samples=2)
    assert isinstance(result, pd.DataFrame)
