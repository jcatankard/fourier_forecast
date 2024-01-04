from fourier_forecast.fourier_forecast import FourierForecast
from statsforecast.models import AutoARIMA, MSTL
from holidays import country_holidays
from datetime import date, timedelta
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from itertools import product
from prophet import Prophet
import pandas as pd
import numpy as np


def find_holidays(markets: list[str], start_date: date, end_date: date) -> dict[str, list[date]]:
    """return dictionary with just holiday dates by market"""
    yrs = range(start_date.year, end_date.year + 1)
    return {m: country_holidays(country=m, years=yrs).keys() for m in markets}


def find_holidays_pdf(markets: list[str], start_date: date, end_date: date) -> pd.DataFrame:
    """return dataframe by date & market with is_holiday column where True for holiday date and false otherwise"""
    mh = find_holidays(markets, start_date, end_date)
    dates = np.arange(start_date, end_date + timedelta(days=1)).astype('O')
    df = pd.DataFrame(product(markets, dates), columns=['MARKET', 'DATE'])
    return df.assign(IS_HOLIDAY=df.apply(lambda z: z['DATE'] in mh[z['MARKET']], axis=1))


def calculate_mape(preds: NDArray, actuals: NDArray) -> float:
    return np.abs(preds / actuals - 1).mean()


def plot_results(test_dates: NDArray, actuals: NDArray, preds: NDArray):
    plt.plot(test_dates, actuals, label='actuals')
    plt.plot(test_dates, preds, label='preds')
    plt.legend()
    plt.show()


if __name__ == '__main__':

    # https://data.cityofevanston.org/dataset/CTA-Ridership-Daily-Boarding-Totals/bnrf-isry
    df = (pd.read_csv('./cta_ridership_totals.csv')
          .assign(ds=lambda x: pd.to_datetime(x['date']).dt.date,
                  y=lambda x: x['total_rides']
                  )
          .sort_values(by='ds', ascending=True)
          )
    df = df[['ds', 'y']]
    hols = find_holidays_pdf(['US'], df['ds'].min(), df['ds'].max())[['IS_HOLIDAY']].values

    train_years = 2
    first_year, last_year = 2003, 2018
    ff_results = np.zeros(last_year - first_year + 1, dtype=np.float64)
    prophet_results = np.zeros(last_year - first_year + 1, dtype=np.float64)
    aa_results = np.zeros(last_year - first_year + 1, dtype=np.float64)
    for i, test_year in enumerate(range(first_year, last_year + 1)):

        train_dates = (date(test_year - train_years, 1, 1), date(test_year - 1, 12, 31))
        test_dates = (date(test_year, 1, 1), date(test_year, 12, 31))

        train_index = df[(df['ds'] >= train_dates[0]) & (df['ds'] <= train_dates[1])].index
        test_index = df[(df['ds'] >= test_dates[0]) & (df['ds'] <= test_dates[1])].index
        h = test_index.size

        train_df = df[df.index.isin(train_index)]
        test_df = df[df.index.isin(test_index)]
        train_hols = hols[train_index]
        test_hols = hols[test_index]

        # FourierForecast ##############
        ff = FourierForecast(weekly_seasonality_terms=3, yearly_seasonality_terms=10)
        y = np.log(train_df['y'].values)
        ff.fit(y, regressors=train_hols)

        preds = ff.predict(h=h, regressors=test_hols)
        preds = np.exp(preds)

        ff_results[i] = calculate_mape(preds, test_df['y'].values)

        # Prophet ##############
        m = Prophet(seasonality_mode='multiplicative', uncertainty_samples=False)
        m.add_country_holidays(country_name='US')
        m.fit(train_df)

        future_df = m.make_future_dataframe(periods=h).tail(h)
        pred_df = m.predict(future_df)

        prophet_results[i] = calculate_mape(pred_df['yhat'].values, test_df['y'].values)

        # AutoArima #############
        # aa = MSTL(season_length=[7, 365], trend_forecaster=AutoARIMA())
        # aa.fit(train_df['y'].values, train_hols.astype(np.float64))
        # aa_preds = aa.predict(h, test_hols.astype(np.float64), level=None)['mean']
        # aa_results[i] = calculate_mape(aa_preds, test_df['y'].values)

    print('FourierForecast:', ff_results)
    print('FourierForecast:', ff_results.mean().round(3))
    print('Prophet:', prophet_results)
    print('Prophet:', prophet_results.mean().round(3))
    # print('AutoArima:', aa_results)
    # print('AutoArima:', aa_results.mean().round(3))
