"""
Find changing effects that indicate hidden confounding.
"""

from ebm_utils.analysis.changepoints import find_discontinuities
from ebm_utils.analysis.changepoints import find_non_monotonicities
from ebm_utils.analysis.plot_utils import plot_feat
from ebm_utils.analysis.plot_utils import standardize


def non_monotonicities(
    adata, outcome_col, ebm_constructor_kwargs=None, ebm_fit_kwargs=None, **kwargs
):
    """Find and plot non-monotoniciites in an AnnData of predictors and outcomes.

    Parameters
    ----------
    adata : AnnData
        AnnData object with observations in rows and features in columns.
    outcome_col : str
        Name of the column in adata.obs that contains the outcome variable.
    ebm_constructor_kwargs : dict, optional
        Keyword arguments to pass to the ExplainableBoostingClassifier constructor.
    ebm_fit_kwargs : dict, optional
        Keyword arguments to pass to the ExplainableBoostingClassifier fit method.
    **kwargs : dict, optional
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
    x = adata.obs
    y = adata.obs[outcome_col]
    x.drop(outcome_col, axis=1, inplace=True)
    results_df, ebm = find_non_monotonicities(
        x,
        y,
        return_ebm=True,
        ebm_constructor_kwargs=kwargs.pop("ebm_constructor_kwargs"),
        ebm_fit_kwargs=kwargs.pop("ebm_fit_kwargs"),
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
    return results_df


def discontinuities(
    adata, outcome_col, ebm_constructor_kwargs=None, ebm_fit_kwargs=None, **kwargs
):
    """Find and plot discontinuities in an AnnData of predictors and outcomes.

    Parameters
    ----------
    adata : AnnData
        AnnData object with observations in rows and features in columns.
    outcome_col : str
        Name of the column in adata.obs that contains the outcome variable.
    ebm_constructor_kwargs : dict, optional
        Keyword arguments to pass to the ExplainableBoostingClassifier constructor.
    ebm_fit_kwargs : dict, optional
        Keyword arguments to pass to the ExplainableBoostingClassifier fit method.
    **kwargs : dict, optional
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
    x = adata.obs
    y = adata.obs[outcome_col]
    x.drop(outcome_col, axis=1, inplace=True)
    results_df, ebm = find_discontinuities(
        x,
        y,
        return_ebm=True,
        ebm_constructor_kwargs=ebm_constructor_kwargs,
        ebm_fit_kwargs=ebm_fit_kwargs,
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
    return results_df
