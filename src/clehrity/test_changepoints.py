import anndata as ad  # type: ignore
import numpy as np
import pandas as pd
import pytest
from pytest import FixtureRequest

from clehrity.changepoints import discontinuities
from clehrity.changepoints import non_monotonicities


# Sample data for testing
N_SAMPLES = 100
SAMPLE_OBS = {
    "feature1": np.random.uniform(-1, 1, size=N_SAMPLES),
    "feature2": np.linspace(-1, 1, num=N_SAMPLES),
    "outcome": [i > 50 for i in range(N_SAMPLES)],
}
SAMPLE_DF = pd.DataFrame(SAMPLE_OBS, dtype=np.float32)
SAMPLE_ANNDATA = ad.AnnData(SAMPLE_DF)


@pytest.fixture
def sample_anndata() -> ad.AnnData:
    """Fixture to provide a sample AnnData object."""
    return SAMPLE_ANNDATA.copy()


def test_non_monotonicities_returns_dataframe(sample_anndata: FixtureRequest) -> None:
    """Test non_monotonicities function returns a DataFrame."""
    result = non_monotonicities(sample_anndata, "outcome")
    assert isinstance(result, pd.DataFrame)


def test_discontinuities_returns_dataframe(sample_anndata: FixtureRequest) -> None:
    """Test discontinuities function returns a DataFrame."""
    result = discontinuities(sample_anndata, "outcome")
    assert isinstance(result, pd.DataFrame)


# More detailed tests could include:
# 1. Testing with various input data to ensure the function handles different cases correctly.
# 2. Validating the structure and content of the returned DataFrame.
# 3. Mocking dependencies if needed to isolate the function's logic.
# 4. Testing edge cases and error handling.
