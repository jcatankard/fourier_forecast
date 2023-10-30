from fourier_forecast.fourier_forecast import FourierForecast
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


def print_mde(preds: NDArray, actuals: NDArray):
    print('----------------')
    print('MAPE:', np.abs(preds / actuals - 1).mean().round(3))
    print('----------------')


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

    train_start, train_end = date(2016, 1, 1), date(2017, 12, 31)
    train_index = df[(df['ds'] >= train_start) & (df['ds'] <= train_end)].index

    test_start = date(2018, 1, 1)
    test_index = df[(df['ds'] >= test_start)].index

    train_df = df[df.index.isin(train_index)]
    test_df = df[df.index.isin(test_index)]
    train_hols = hols[train_index]
    test_hols = hols[test_index]

    ff = FourierForecast(weekly_seasonality_terms=3, yearly_seasonality_terms=10)
    multiplicative = True
    y = np.log(train_df['y'].values) if multiplicative else train_df['y'].values
    ff.fit(train_df['ds'].values, y, regressors=train_hols)

    preds = ff.predict(ds=test_df['ds'].values, regressors=test_hols)
    preds = np.exp(preds) if multiplicative else preds

    ff.plot_components()
    print_mde(preds, test_df['y'].values)
    plot_results(test_df['ds'].values, test_df['y'].values, preds)

    m = Prophet(seasonality_mode='multiplicative')
    m.add_country_holidays(country_name='US')
    m.fit(train_df)

    future_df = m.make_future_dataframe(periods=test_index.size).tail(test_index.size)
    pred_df = m.predict(future_df)

    print_mde(pred_df['yhat'].values, test_df['y'].values)
    plot_results(test_df['ds'].values, test_df['y'].values, pred_df['yhat'].values)
