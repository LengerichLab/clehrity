"""Change point analysis via Explainable Boosting Machines.

Find changing effects that indicate hidden confounding.
Uses the utilities in ebm_utils to find and plot non-monotonicities and discontinuities.
"""

from typing import Any

import anndata as ad  # type: ignore
import numpy as np
import pandas as pd
from ebm_utils.analysis.changepoints import find_discontinuities  # type: ignore
from ebm_utils.analysis.changepoints import find_non_monotonicities
from ebm_utils.analysis.plot_utils import plot_feat  # type: ignore
from ebm_utils.analysis.plot_utils import standardize


def non_monotonicities(
    adata: ad.AnnData, outcome_col: str, **kwargs: Any
) -> pd.DataFrame:
    """Find and plot non-monotoniciites in an AnnData of predictors and outcomes.

    Parameters
    ----------
    adata : AnnData
        AnnData object with observations in rows and features in columns.
    outcome_col : str
        Name of the column in adata.obs that contains the outcome variable.
    kwargs : dict, optional
        Keyword arguments to pass to find_non_monotonicities.

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with the following columns:
            - Feature: name of the feature
            - Value: value of the feature at which the non-monotonicity occurs
            - Direction: direction of the non-monotonicity
            - Score: score of the non-monotonicity
    """
    x = pd.DataFrame(adata.X, columns=adata.var.index).astype(np.float32)
    y = x[outcome_col].copy()
    x.drop(outcome_col, axis=1, inplace=True)
    results_df, ebm = find_non_monotonicities(
        x,
        y,
        return_ebm=True,
        ebm_constructor_kwargs=kwargs.pop("ebm_constructor_kwargs", {}),
        ebm_fit_kwargs=kwargs.pop("ebm_fit_kwargs", {}),
        **kwargs,
    )
    for feature in set(results_df["Feature"].values):
        idxs = results_df["Feature"] == feature
        x_vals = results_df["Value"].loc[idxs].values
        plot_feat(
            ebm.explain_global(),
            feature,
            X_train=x,
            classification=kwargs.get("classification", True),
            axlines={standardize(feature): x_vals},
        )
    return results_df  # type: ignore


def discontinuities(adata: ad.AnnData, outcome_col: str, **kwargs: Any) -> pd.DataFrame:
    """Find and plot discontinuities in an AnnData of predictors and outcomes.

    Parameters
    ----------
    adata : AnnData
        AnnData object with observations in rows and features in columns.
    outcome_col : str
        Name of the column in adata.obs that contains the outcome variable.
    kwargs : dict, optional
        Keyword arguments to pass to find_discontinuities.

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with the following columns:
            - Feature: name of the feature
            - Value: value of the feature at which the discontinuity occurs
            - Direction: direction of the discontinuity
            - Score: score of the discontinuity
    """
    x = pd.DataFrame(adata.X, columns=adata.var.index).astype(np.float32)
    y = x[outcome_col].copy()
    x.drop(outcome_col, axis=1, inplace=True)
    results_df, ebm = find_discontinuities(
        x,
        y,
        return_ebm=True,
        ebm_constructor_kwargs=kwargs.pop("ebm_constructor_kwargs", {}),
        ebm_fit_kwargs=kwargs.pop("ebm_fit_kwargs", {}),
        **kwargs,
    )
    for feature in set(results_df["Feature"].values):
        idxs = results_df["Feature"] == feature
        x_vals = results_df["Value"].loc[idxs].values
        plot_feat(
            ebm.explain_global(),
            feature,
            X_train=x,
            classification=kwargs.get("classification", True),
            axlines={standardize(feature): x_vals},
        )
    return results_df  # type: ignore
