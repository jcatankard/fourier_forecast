import matplotlib.pyplot as plt
from fourier_forecast import FourierForecast
import pandas as pd
import numpy as np
import datetime


def create_signal(n: int) -> pd.DataFrame:
    # create time
    dt = 1
    t = np.arange(0, n, dt)

    # create signal
    weekly_seasonality = 2 * np.cos(2 * np.pi * t / 7) + 7
    monthly_seasonality = 1 * np.sin(2 * np.pi * t / 30.43) + 12
    quarterly_seasonality = 1 * np.sin(2 * np.pi * t / 91.31) + 4
    yearly_seasonality = 5 * np.sin(2 * np.pi * t / 365.25) + 1
    trend = 20 * np.arange(n) / n
    y_clean = weekly_seasonality + yearly_seasonality + trend + quarterly_seasonality + monthly_seasonality
    y = y_clean + 0.5 * np.random.randn(n)

    df = pd.DataFrame()
    df['ds'] = pd.date_range(start=datetime.date(2020, 1, 1), periods=n)
    df['y'] = y

    return df


# fit
train_len = 731
test_len = 730
df = create_signal(train_len + test_len)
model = FourierForecast()
model.fit(df.head(train_len))
print(model.coefs)

# predict
df['pred'] = model.predict(df.drop(columns=['y']))
rmse = np.mean((df['pred'] - df['y']) ** 2) ** 0.5
print('rmse', rmse)

plt.plot(df['ds'], df['y'], label='y')
plt.plot(df['ds'][: train_len], df['pred'][: train_len], label='train')
plt.plot(df['ds'], np.where(df.index < train_len, np.nan, df['pred']), label='pred')

plt.plot(df['ds'][: train_len], model.seasonality, label='seasonality')
plt.plot(df['ds'][: train_len], model.trend, label='trend')
plt.plot(df['ds'][: train_len], model.noise, label='noise')

plt.legend()
plt.show()
