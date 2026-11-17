"""LightGBM baseline: Tweedie objective, lags 1..26, rolling mean/std/max/min over 4/13/26/52 weeks plus an expanding mean of
lag 1, week/month/quarter calendar features. Trained with the direct strategy (one model per lead, max_horizon=13) by the runner."""
import lightgbm as lgb
from mlforecast import MLForecast
from mlforecast.lag_transforms import ExpandingMean, RollingMax, RollingMean, RollingMin, RollingStd

WINDOWS = (4, 13, 26, 52)


def make_lgbm(seed: int, freq: str = "W-SAT") -> MLForecast:
    tf = [t(window_size=w) for w in WINDOWS for t in (RollingMean, RollingStd, RollingMax, RollingMin)] + [ExpandingMean()]
    model = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.5, n_estimators=200, num_leaves=127,
                              random_state=seed, verbose=-1)
    return MLForecast(models={"yhat": model}, freq=freq, lags=list(range(1, 27)), lag_transforms={1: tf},
                      date_features=["week", "month", "quarter"])
