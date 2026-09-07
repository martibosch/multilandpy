"""Scale-of-effect evaluation."""

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy import typing as npt
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted, validate_data
from statsmodels.base.model import Model

from focalpy import settings, utils

__all__ = [
    "ScaleOfEffectSelector",
    "scale_eval_ser",
    "scale_of_effect_features",
]


def _scipy_statistic(
    X: npt.ArrayLike,
    y: npt.ArrayLike,
    func: Callable,
    *args: Sequence,
    **kwargs: utils.KwargsType,
) -> float:
    # raise for stats.ConstantInputWarning
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # TODO: do we need a try/except? if so, which error to catch?
        # try:
        return func(X, y, **kwargs).statistic
        # except:
        #     return np.nan


def _model_attr(
    X: npt.ArrayLike,
    y: npt.ArrayLike,
    model: Model,
    attr: str,
    *,
    add_constant: bool = True,
    **model_kwargs: utils.KwargsType,
) -> float:
    """
    Extract an attribute from a (fitted) model (statsmodels or spreg).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target vector.
    model : Model
        A statsmodels or spreg model class.
    attr : str
        The attribute to extract from the model.
    add_constant : bool, default True

    Returns
    -------
    float
        The value of the specified attribute.
    """
    if add_constant:
        X = sm.add_constant(X)
    _model = model(y, X, **model_kwargs)
    if hasattr(_model, "fit"):
        # compat between statsmodels and spreg
        _model = _model.fit()
    return getattr(_model, attr)


def scale_eval_ser(
    X_df: pd.DataFrame,
    y_ser: pd.Series,
    *,
    how: str | None = None,
    criteria: str | None = None,
    model: Model | None = None,
    **eval_func_kwargs: utils.KwargsType,
) -> pd.Series:
    """
    Evaluate the scale effect of `X_df` against `y_ser`.

    Parameters
    ----------
    X_df : pandas.DataFrame
        Feature data frame where each column represents a feature at a specific scale,
        following the naming pattern "{feature}_{scale}" (e.g., `"density_500"`).
    y_ser : pandas.Series
        Response variable, as pandas Series with the same index as `X_df`.
    criteria : str, optional
        The evaluation criteria to use, which can be either a statistical test from
        `scipy.stats` (e.g., `"pearsonr"`, `"spearmanr"`) or a model attribute from a
        statsmodels or spreg model (e.g., `"rsquared"`, `"aic"`). If `None`, defaults
        to `settings.SCALE_OF_EFFECT_CRITERIA`.
    model : statsmodels or spreg Model class, optional
        The model class to use if `criteria` is a model attribute. If `None`, defaults
        to `settings.SCALE_OF_EFFECT_MODEL`. Ignored if `criteria` is a `scipy.stats`
        function.
    **eval_func_kwargs : mapping, optional
        Keyword arguments to pass to the evaluation function.

    Returns
    -------
    pandas.Series
        A Series with MultiIndex (feature group, scale) containing the evaluation scores
        for each feature at each scale.
    """
    # process criteria arg
    if criteria is None:
        criteria = settings.SCALE_OF_EFFECT_CRITERIA
    if hasattr(stats, criteria):
        # non-parametric with scipy.stats
        eval_func = _scipy_statistic
        scipy_func = getattr(stats, criteria)
        extra_eval_func_args = [scipy_func]
    else:
        # parametric with statsmodels or spreg
        eval_func = _model_attr
        if model is None:
            model = settings.SCALE_OF_EFFECT_MODEL
        extra_eval_func_args = [model, criteria]

    if how is None:
        how = settings.SCALE_OF_EFFECT_HOW
    if how == "individual":

        def process_by(split_cols):
            return split_cols.str[:-1].map(lambda col_parts: "_".join(col_parts))

        def group_apply(group_df):
            return group_df.T.apply(eval_func, args=(y_ser, *extra_eval_func_args))
    else:  # "global"/"all"
        if hasattr(stats, criteria):
            # functions such as `scipy.stats.spearmanr` are not designed for
            # multivariate inputs, so we cannot do global evaluation with them (unless
            # we accepted an additional aggregation step, e.g., mean of correlations)
            raise ValueError(
                "Global scale evaluation is not supported for `scipy.stats` functions."
            )

        def process_by(split_cols):
            return split_cols.str[-1]

        def group_apply(group_df):
            return eval_func(group_df.T, y_ser, *extra_eval_func_args)

    return (
        X_df.T.groupby(by=process_by(X_df.columns.str.split("_")))
        .apply(group_apply)
        .rename(criteria)
    )


def scale_of_effect_features(
    X_df: pd.DataFrame,
    y_ser: pd.Series,
    *,
    how: str | None = None,
    criteria: str | None = None,
    direction: str | None = None,
    model: Model | None = None,
    **eval_func_kwargs: utils.KwargsType,
) -> np.ndarray:
    """
    Identify the scale-of-effect for each feature in `X_df` against `y_ser`.

    Parameters
    ----------
    X_df : pandas.DataFrame
        Feature data frame where each column represents a feature at a specific scale,
        following the naming pattern "{feature}_{scale}" (e.g., `"density_500"`).
    y_ser : pandas.Series
        Response variable, as pandas Series with the same index as `X_df`.
    criteria : str, optional
        The evaluation criteria to use, which can be either a statistical test from
        `scipy.stats` (e.g., `"pearsonr"`, `"spearmanr"`) or a model attribute from a
        statsmodels or spreg model (e.g., `"rsquared"`, `"aic"`). If `None`, defaults
        to `settings.SCALE_OF_EFFECT_CRITERIA`.
    direction : {"max", "min", "absmax"}, optional
        The direction of the criteria: `"max"` (higher is better), `"min"`
        (lower is better), or `"absmax"` (strongest absolute value, suitable for
        correlation coefficients that can be negative). If `None`, the direction is
        inferred from `settings.SCALE_OF_EFFECT_CRITERIA_DIRECTION_DICT`. Required if
        the direction cannot be inferred.
    model : statsmodels or spreg Model class, optional
        The model class to use if `criteria` is a model attribute. If `None`, defaults
        to `settings.SCALE_OF_EFFECT_MODEL`. Ignored if `criteria` is a `scipy.stats`
        function.
    **eval_func_kwargs : mapping, optional
        Keyword arguments to pass to the evaluation function.

    Returns
    -------
    numpy.ndarray
        An array of scale-of-effect feature names for each feature in `X_df`.
    """
    if criteria is None:
        # ACHTUNG: we are doing the if None to get the default criteria from the
        # settings both in `scale_eval_ser` and `scale_of_effect`. This is not ideal but
        # serves to avoid computing the scale evaluation if there is no direction for
        # the selected criteria
        criteria = settings.SCALE_OF_EFFECT_CRITERIA
    if direction is None:
        direction = settings.SCALE_OF_EFFECT_CRITERIA_DIRECTION_DICT.get(criteria, None)
    if direction is None:
        raise ValueError(
            f"Direction (min/max) for criteria '{criteria}' is not specified and not "
            "found in settings."
        )

    eval_ser = scale_eval_ser(
        X_df,
        y_ser,
        how=how,
        criteria=criteria,
        model=model,
        **eval_func_kwargs,
    )

    # absmax: select by strongest absolute value (for signed criteria like
    # correlation coefficients where negative values indicate strong inverse
    # relationships, not weak ones)
    if direction == "absmax":
        eval_ser = eval_ser.abs()
        direction = "max"

    if isinstance(eval_ser.index, pd.MultiIndex):
        # how == "individual"
        # TODO: manage index level names better than default "level_0", "level_1", etc?
        eval_ser_gb = eval_ser.reset_index(level=0).groupby("level_0")
        return getattr(eval_ser_gb, f"idx{direction}")().values.flatten()
    else:
        # how == "global"
        scale_of_effect = getattr(eval_ser, f"idx{direction}")()
        return X_df.columns[X_df.columns.str.split("_").str[-1] == scale_of_effect]


class ScaleOfEffectSelector(SelectorMixin, BaseEstimator):
    """Select features at their optimal spatial scale.

    For each feature group (identified by the column naming convention
    ``{feature}_{scale}``), selects the column at the scale that optimizes
    the given criteria against the target variable. Designed for use in a
    scikit-learn :class:`~sklearn.pipeline.Pipeline`, ensuring that scale
    selection is performed within each cross-validation fold.

    Parameters
    ----------
    how : {"individual", "global"}, optional
        How to evaluate scales. ``"individual"`` evaluates each feature group
        independently (univariate), ``"global"`` evaluates all features jointly
        at each scale. If ``None``, defaults to
        :data:`~focalpy.settings.SCALE_OF_EFFECT_HOW`.
    criteria : str, optional
        Evaluation criteria. Can be a ``scipy.stats`` function name
        (e.g., ``"spearmanr"``, ``"pearsonr"``) or a model attribute
        (e.g., ``"rsquared"``, ``"aic"``). If ``None``, defaults to
        :data:`~focalpy.settings.SCALE_OF_EFFECT_CRITERIA`.
    direction : {"max", "min", "absmax"}, optional
        Whether higher (``"max"``), lower (``"min"``), or strongest absolute
        (``"absmax"``) values of the criteria indicate a better fit. Use
        ``"absmax"`` for signed criteria like correlation coefficients. If
        ``None``, inferred from
        :data:`~focalpy.settings.SCALE_OF_EFFECT_CRITERIA_DIRECTION_DICT`.
    model : statsmodels or spreg Model class, optional
        Model class for parametric criteria. Ignored for ``scipy.stats``
        criteria. If ``None``, defaults to
        :data:`~focalpy.settings.SCALE_OF_EFFECT_MODEL`.

    Attributes
    ----------
    selected_features_ : numpy.ndarray
        Column names of the selected features (one per feature group).
    n_features_in_ : int
        Number of features seen during :meth:`fit`.
    feature_names_in_ : numpy.ndarray
        Feature names seen during :meth:`fit`.

    Examples
    --------
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.linear_model import LinearRegression
    >>> pipe = Pipeline(
    ...     [
    ...         ("soe", ScaleOfEffectSelector(criteria="spearmanr", how="individual")),
    ...         ("scaler", StandardScaler()),
    ...         ("model", LinearRegression()),
    ...     ]
    ... )
    >>> pipe.fit(X_df, y)  # doctest: +SKIP

    See Also
    --------
    scale_of_effect_features : Underlying feature selection function.
    scale_eval_ser : Scale evaluation scoring function.
    """

    def __init__(self, how=None, criteria=None, direction=None, model=None):
        self.how = how
        self.criteria = criteria
        self.direction = direction
        self.model = model

    def fit(self, X, y):
        """Determine the optimal scale for each feature group.

        Parameters
        ----------
        X : {array-like, DataFrame} of shape (n_samples, n_features)
            Feature matrix. Columns must follow the ``{feature}_{scale}``
            naming convention. Passing a :class:`~pandas.DataFrame` is
            recommended; numpy arrays require that a previous pipeline step
            has set ``feature_names_in_``.
        y : array-like of shape (n_samples,)
            Target variable.

        Returns
        -------
        self
            Fitted selector.
        """
        # _validate_data sets n_features_in_ and feature_names_in_ (if X is a
        # DataFrame)
        X = validate_data(self, X, dtype=None)

        if isinstance(X, pd.DataFrame):
            X_df = X
        elif hasattr(self, "feature_names_in_"):
            X_df = pd.DataFrame(np.asarray(X), columns=self.feature_names_in_)
        else:
            raise ValueError(
                f"{self.__class__.__name__} requires named features following "
                "the '{{feature}}_{{scale}}' convention. Pass X as a pandas "
                "DataFrame."
            )

        y_ser = y if isinstance(y, pd.Series) else pd.Series(np.asarray(y))

        # only pass non-None params so scale_of_effect_features uses its defaults
        kwargs = {
            param: val
            for param in ("how", "criteria", "direction", "model")
            if (val := getattr(self, param)) is not None
        }

        self.selected_features_ = scale_of_effect_features(X_df, y_ser, **kwargs)
        self.support_mask_ = np.isin(self.feature_names_in_, self.selected_features_)
        return self

    def _get_support_mask(self):
        check_is_fitted(self, "support_mask_")
        return self.support_mask_
