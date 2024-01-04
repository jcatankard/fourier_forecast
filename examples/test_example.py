from fourier_forecast.fourier_forecast import FourierForecast
from tests.create_data import create_data
import plotly.graph_objs as go
import numpy as np
import time


if __name__ == '__main__':

    log_y = True
    noise_scale = 500 if log_y else 2.5
    fourier_terms = np.array([3, 0, 0, 10])

    ds, clean, regressors = create_data(regressors=True, fourier_terms=fourier_terms, log_y=log_y)
    actuals = clean + np.random.normal(0, 1, ds.size) * noise_scale

    ff = FourierForecast(log_y=log_y, *fourier_terms)
    n = 700
    start = time.time()
    ff.fit(actuals[: n], regressors=regressors[: n])
    print('Time to fit:', round(time.time() - start, 3))
    ff.plot_components().show()
    # ff.plot_seasonality_components().show()
    # ff.plot_regressor_components().show()

    pred = ff.predict(h=ds.size - n, regressors=regressors[n:])

    print('PREDS VS ACTUALS:', np.abs(pred - actuals[n:]).mean().round(3))
    print('FITTED vs ACTUALS:', np.abs(ff.fitted() - actuals[: n]).mean().round(3))
    print('CLEAN vs ACTUALS:', np.abs(clean[: n] - actuals[: n]).mean().round(3))
    print('PREDS vs CLEAN:', np.abs(pred - clean[n:]).mean().round(3))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ds, y=actuals, mode='lines', name='actuals'))
    fig.add_trace(go.Scatter(x=ds[: n], y=ff.fitted(), mode='lines', name='fitted'))
    fig.add_trace(go.Scatter(x=ds[n:], y=pred, mode='lines', name='preds'))
    fig.update_layout(showlegend=True)
    fig.show()
