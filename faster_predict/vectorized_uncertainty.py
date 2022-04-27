import numpy as np
import pandas as pd

from prophet import Prophet


def _make_historical_mat_time(deltas, changepoints_t, t_time, n_row=1):
    """
    Creates a matrix of slope-deltas where these changes occured in training data according to the trained prophet obj
    """
    diff = np.diff(t_time).mean()
    prev_time = np.arange(0, 1 + diff, diff)
    idxs = []
    for changepoint in changepoints_t:
        idxs.append(np.where(prev_time > changepoint)[0][0])
    prev_deltas = np.zeros(len(prev_time))
    prev_deltas[idxs] = deltas
    prev_deltas = np.repeat(prev_deltas.reshape(1, -1), n_row, axis=0)
    return prev_deltas, prev_time


def prophet_logistic_uncertainty(
    mat: np.ndarray,
    deltas: np.ndarray,
    prophet_obj: Prophet,
    cap_scaled: np.ndarray,
    t_time: np.ndarray,
):
    """
    Vectorizes prophet's logistic growth uncertainty by creating a matrix of future possible trends.
    """

    def ffill(arr):
        mask = arr == 0
        idx = np.where(~mask, np.arange(mask.shape[1]), 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        return arr[np.arange(idx.shape[0])[:, None], idx]

    k = prophet_obj.params["k"][0]
    m = prophet_obj.params["m"][0]
    n_length = len(t_time)
    #  for logistic growth we need to evaluate the trend all the way from the start of the train item
    historical_mat, historical_time = _make_historical_mat_time(deltas, prophet_obj.changepoints_t, t_time, len(mat))
    mat = np.concatenate([historical_mat, mat], axis=1)
    full_t_time = np.concatenate([historical_time, t_time])

    #  apply logistic growth logic on the slope changes
    k_cum = np.concatenate((np.ones((mat.shape[0], 1)) * k, np.where(mat, np.cumsum(mat, axis=1) + k, 0)), axis=1)
    k_cum_b = ffill(k_cum)
    gammas = np.zeros_like(mat)
    for i in range(mat.shape[1]):
        x = full_t_time[i] - m - np.sum(gammas[:, :i], axis=1)
        ks = 1 - k_cum_b[:, i] / k_cum_b[:, i + 1]
        gammas[:, i] = x * ks
    # the data before the -n_length is the historical values, which are not needed, so cut the last n_length
    k_t = (mat.cumsum(axis=1) + k)[:, -n_length:]
    m_t = (gammas.cumsum(axis=1) + m)[:, -n_length:]
    sample_trends = cap_scaled / (1 + np.exp(-k_t * (t_time - m_t)))
    # remove the mean because we only need width of the uncertainty centered around 0
    # we will add the width to the main forecast - yhat (which is the mean) - later
    sample_trends = sample_trends - sample_trends.mean(axis=0)
    return sample_trends


def _make_trend_shift_matrix(mean_delta: float, likelihood: float, future_length: float, k: int = 10000) -> np.ndarray:
    """
    Creates a matrix of random trend shifts based on historical likelihood and size of shifts.
    Can be used for either linear or logistic trend shifts.
    Each row represents a different sample of a possible future, and each column is a time step into the future.
    """
    # create a bool matrix of where these trend shifts should go
    bool_slope_change = np.random.uniform(size=(k, future_length)) < likelihood
    shift_values = np.random.laplace(0, mean_delta, size=bool_slope_change.shape)
    mat = shift_values * bool_slope_change
    n_mat = np.hstack([np.zeros((len(mat), 1)), mat])[:, :-1]
    mat = (n_mat + mat) / 2
    return mat


def add_prophet_uncertainty(
    prophet_obj: Prophet,
    forecast_df: pd.DataFrame,
    using_train_df: bool = False,
):
    """
    Adds yhat_upper and yhat_lower to the forecast_df used by fbprophet, based on the params of a trained prophet_obj
    and the interval_width.
    Use using_train_df=True if the forecast_df is not for a future time but for the training data.
    """
    assert prophet_obj.history is not None, "Model has not been fit"
    assert "yhat" in forecast_df.columns, "Must have the mean yhat forecast to build uncertainty on"
    interval_width = prophet_obj.interval_width

    if using_train_df:  # there is no trend-based uncertainty if we're only looking on the past where trend is known
        sample_trends = np.zeros(10000, len(forecast_df))
    else:  # create samples of possible future trends
        future_time_series = ((forecast_df["ds"] - prophet_obj.start) / prophet_obj.t_scale).values
        single_diff = np.diff(future_time_series).mean()
        change_likelihood = len(prophet_obj.changepoints_t) * single_diff
        deltas = prophet_obj.params["delta"][0]
        n_length = len(forecast_df)
        mean_delta = np.mean(np.abs(deltas)) + 1e-8
        if prophet_obj.growth == "linear":
            mat = _make_trend_shift_matrix(mean_delta, change_likelihood, n_length, k=10000)
            sample_trends = mat.cumsum(axis=1).cumsum(axis=1)  # from slope changes to actual values
            sample_trends = sample_trends * single_diff  # scaled by the actual meaning of the slope
        elif prophet_obj.growth == "logistic":
            mat = _make_trend_shift_matrix(mean_delta, change_likelihood, n_length, k=1000)
            cap_scaled = (forecast_df["cap"] / prophet_obj.y_scale).values
            sample_trends = prophet_logistic_uncertainty(mat, deltas, prophet_obj, cap_scaled, future_time_series)
        else:
            raise NotImplementedError

    # add gaussian noise based on historical levels
    sigma = prophet_obj.params["sigma_obs"][0]
    historical_variance = np.random.normal(scale=sigma, size=sample_trends.shape)
    full_samples = sample_trends + historical_variance
    # get quantiles and scale back (prophet scales the data before fitting, so sigma and deltas are scaled)
    width_split = (1 - interval_width) / 2
    quantiles = np.array([width_split, 1 - width_split]) * 100  # get quantiles from width
    quantiles = np.percentile(full_samples, quantiles, axis=0)
    # Prophet scales all the data before fitting and predicting, y_scale re-scales it to original values
    quantiles = quantiles * prophet_obj.y_scale

    forecast_df["yhat_lower"] = forecast_df.yhat + quantiles[0]
    forecast_df["yhat_upper"] = forecast_df.yhat + quantiles[1]


if __name__ == '__main__':
    # example usage

    # create a time series (or load one)
    import datetime
    n = 100
    training_df = pd.DataFrame(
        zip(*[np.cumsum(np.random.rand(n) - 0.45),
              pd.date_range(datetime.datetime(2020, 1, 1), freq='w', periods=n)]),
        columns=['df', 'y'])

    # tell Prophet not to create the interval by itself by uncertainty_samples=None
    p = Prophet(uncertainty_samples=None)
    p = p.fit(training_df)

    # no need to run this part if you only want to forecast the future
    training_df = p.predict(training_df)
    add_prophet_uncertainty(p, training_df, using_train_df=True)

    # set to your number of periods and freq
    forecast_df = p.make_future_dataframe(periods=10, freq='W', include_history=False)
    forecast_df = p.predict(forecast_df)
    add_prophet_uncertainty(p, forecast_df)
    # training_df and forecast_df will now include the confidence interval
