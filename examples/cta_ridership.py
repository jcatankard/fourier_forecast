from fourier_forecast.fourier_forecast import FourierForecast
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


if __name__ == '__main__':

    # https://data.cityofevanston.org/dataset/CTA-Ridership-Daily-Boarding-Totals/bnrf-isry
    df = pd.read_csv('./cta_ridership_totals.csv').tail(366 * 3)
    ds = pd.to_datetime(df['date']).dt.date.values
    actuals = df['total_rides'].values

    train_len = 366 * 2
    train_actuals = actuals[: train_len]
    train_ds = ds[: train_len]
    test_actuals = actuals[train_len:]
    test_ds = ds[train_len:]

    ff = FourierForecast(weekly_seasonality_terms=3, yearly_seasonality_terms=10)

    multiplicative = True

    train_actuals = np.log(train_actuals) if multiplicative else train_actuals
    ff.fit(train_ds, train_actuals)

    preds = ff.predict(ds=test_ds)
    preds = np.exp(preds) if multiplicative else preds

    ff.plot_components()

    print('----------------')
    print('MAPE:', np.abs(preds / test_actuals - 1).mean().round(3))

    plt.plot(test_ds, test_actuals, label='actuals')
    plt.plot(test_ds, preds, label='preds')
    plt.legend()
    plt.show()
