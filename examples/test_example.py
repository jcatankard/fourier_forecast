from fourier_forecast.fourier_forecast import FourierForecast
from tests.create_data import create_data
import matplotlib.pyplot as plt
import numpy as np
import time


if __name__ == '__main__':
    fourier_terms = np.array([3, 0, 0, 10])
    ds, clean, regressors = create_data(regressors=True, fourier_terms=fourier_terms)
    actuals = clean + np.random.normal(0, 1, ds.size) * 2.5

    ff = FourierForecast(*fourier_terms)
    n = 700
    start = time.time()
    ff.fit(actuals[: n], regressors=regressors[: n])
    print('Time to fit:', round(time.time() - start, 3))
    ff.plot_components()

    pred = ff.predict(h=ds.size - n, regressors=regressors[n:])

    print('PREDS VS ACTUALS:', np.abs(pred - actuals[n:]).mean().round(3))
    print('FITTED vs ACTUALS:', np.abs(ff.fitted() - actuals[: n]).mean().round(3))
    print('CLEAN vs ACTUALS:', np.abs(clean[: n] - actuals[: n]).mean().round(3))
    print('PREDS vs CLEAN:', np.abs(pred - clean[n:]).mean().round(3))

    plt.plot(ds, actuals, label='actuals')
    plt.plot(ds[: n], ff.fitted(), label='fitted')
    plt.plot(ds[n:], pred, label='preds')
    # plt.plot(ds, clean, label='clean')
    plt.legend()
    plt.show()
