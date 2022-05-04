import numpy as np
import pandas as pd
import time
import datetime
from prophet import Prophet
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from batch_elastic_net import BatchElasticNetRegression


def make_sine_wave(length: int, n_cycles: int):
    """
    Makes a sine wave given some length and the number of cycles it should go through in that period
    """
    samples = np.linspace(0, length, length)
    return np.sin(2 * np.pi * n_cycles * samples)


def generate_dataset(n_items):
    """
    Generates a time series dataset with weekly frequency for two years. Randomly assigns the yearly, monthly and
    trend values for each item
    """
    year_in_weeks = 104
    yealy_s = make_sine_wave(year_in_weeks, 2)
    monthly_s = make_sine_wave(year_in_weeks, year_in_weeks / 24)
    trend = np.arange(year_in_weeks) / year_in_weeks
    all_ys = []
    for i in range(n_items):
        d = (np.stack([yealy_s, monthly_s, trend], axis=1) * np.random.rand(3)).sum(axis=1) + np.random.rand(year_in_weeks)
        all_ys.append(d + (np.random.rand(len(d))-0.45).cumsum())
    return pd.DataFrame(zip(*all_ys), index = pd.date_range(datetime.datetime(2020, 1, 1), freq='w', periods=len(d)))


def get_changepoint_idx(length, n_changepoints, changepoint_range=0.8):
    """
    Finds the indices of slope change-points using Prophet's logic: assign them uniformly of the first changepoint_range
    percentage of the data
    """
    hist_size = int(np.floor(length * changepoint_range))
    return np.linspace(0, hist_size - 1, n_changepoints+1).round().astype(int)[1:]


def make_changepoint_features(n, changes_idx):
    """
    Creates initial slope and slope change-points features given a length of data and locations of indices.
    The features are 0s for the first elements until their idx is reached, and then they move linearly upwards.
    These features can be used to model a time series with an initial slope and the deltas of change-points.
    """
    linear = np.arange(n).reshape(-1,1)
    feats = [linear]
    for i in changes_idx:
        slope_feat = np.zeros(n)
        slope_feat[i:] = np.arange(0, n-i)
        slope_feat = slope_feat.reshape(-1,1)
        feats.append(slope_feat)
    feat = np.concatenate(feats, axis=1)
    return feat


def run_prophet():
    t = time.time()
    all_prophets_datasets_forecasts = {}
    for name, dataset in data_sets.items():
        all_p_forecast = []
        for i in range(dataset.shape[1]):
            ds = dataset.iloc[:, i].reset_index()
            ds.columns = ['ds', 'y']
            # if uncertainty samples is not None it will take way more time
            m = Prophet(n_changepoints=n_changepoints, changepoint_prior_scale=change_prior, growth='linear',
                        uncertainty_samples=None,
                        yearly_seasonality=True, weekly_seasonality=False, seasonality_prior_scale=seasonality_prior)
            m.fit(ds)
            forecast = m.predict(ds)
            all_p_forecast.append(forecast.yhat)
        all_prophets_datasets_forecasts[name] = pd.DataFrame(zip(*all_p_forecast), index=ds.ds)

    return all_prophets_datasets_forecasts, time.time() - t


def run_batch_linear():
    big_num = 20.  # used as std of prior when it should be uninformative
    p = Prophet()
    t = time.time()
    all_BatchLinear_datasets_forecasts = {}
    for name, dataset in data_sets.items():
        dates = pd.Series(dataset.index)
        dataset_length = len(dataset)
        idx = get_changepoint_idx(dataset_length, n_changepoints)

        seasonal_feat = p.make_seasonality_features(dates, 365.25, 10, 'yearly_sine')
        changepoint_feat = make_changepoint_features(dataset_length, idx) / dataset_length
        feat = np.concatenate([changepoint_feat, seasonal_feat], axis=1)

        n_changepoint_feat = changepoint_feat.shape[1] - 1
        # laplace prior only on changepoints (seasonals get big_num, to avoid l1 regularization on it)
        l1_priors = np.array([big_num] + [change_prior] * n_changepoint_feat + [big_num] * seasonal_feat.shape[1])
        # normal prior on initial slope and on seasonals, and a big_num on changepoints to avoid l2 regularization
        l2_priors = np.array([5] + [big_num] * n_changepoint_feat + [seasonality_prior] * seasonal_feat.shape[
            1])  # normal prior only on seasonal

        # this is how Prophet scales the data before fitting - divide by max of each item
        scale = dataset.max()
        scaled_y = dataset / scale

        blr = BatchElasticNetRegression()
        blr.fit(feat, scaled_y, l1_reg_params=l1_priors, l2_reg_params=l2_priors, as_bayesian_prior=True, verbose=True,
                iterations=1500)

        # get the predictions for the train
        all_BatchLinear_datasets_forecasts[name] = pd.DataFrame(blr.predict(feat) * scale.values, index=dates)

    return all_BatchLinear_datasets_forecasts, time.time() - t


if __name__ == '__main__':
    data_files_names = ['d1', 'd2', 'M5_sample']
    data_sets = {name: pd.read_csv(f'data_files/{name}.csv', index_col=0, parse_dates=True) for name in data_files_names}
    data_sets['randomly_generated'] = generate_dataset(500)

    # can play with these params for both predictors
    change_prior = 0.5
    # the seasonality_prior is an uninformative prior (hardly any regularization), which is the default for Prophet and usually does not require changing
    seasonality_prior = 10
    n_changepoints = 15

    all_prophets_datasets_forecasts, prophet_time = run_prophet()
    all_BatchLinear_datasets_forecasts, batch_time = run_batch_linear()

    print(f'total number of items: {sum([x.shape[1] for x in data_sets.values()])}')
    print(f'Prophet time: {round(prophet_time, 2)}; batch time: {round(batch_time, 2)}')

    # plot examples from datasets (copy to notebook and repeat for different items and datasets)
    name = 'd1'
    batch_preds = all_BatchLinear_datasets_forecasts[name]
    prophet_preds = all_prophets_datasets_forecasts[name]
    orig_data = data_sets[name]

    i = np.random.randint(0, orig_data.shape[1])
    orig_data.iloc[:, i].plot(label='target')
    batch_pred = batch_preds.iloc[:, i]
    prophet_pred = prophet_preds.iloc[:, i]
    prophet_pred.plot(label='prophet')
    batch_pred.plot(label='my_batch')
    plt.title(f'Pearson {round(pearsonr(batch_pred, prophet_pred)[0], 3)}')
    plt.legend()
    plt.show()

    # mean pearson
    all_corrs = {}
    for name in data_sets.keys():
        batch_preds = all_BatchLinear_datasets_forecasts[name]
        prophet_preds = all_prophets_datasets_forecasts[name]
        corrs = []
        for i in range(prophet_preds.shape[1]):
            corrs.append(pearsonr(batch_preds.iloc[:, i], prophet_preds.iloc[:, i])[0])
        all_corrs[name] = np.mean(corrs)
    print(all_corrs)



