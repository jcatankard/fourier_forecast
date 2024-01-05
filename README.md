# FourierForecast
A time-series modelling approach that decomposes a daily time-series into seasonality, trend, bias,
optional regressors and noise.
 
- Seasonality types & Fourier terms are specified
- Seasonality components, trends, bias and regressors are fit with gradient descent
- Components can be visualised with plot_components method
- Numba is used for performance boost

### Blog posts:
 - [Part I](https://medium.com/@jcatankard_76170/forecasting-with-fourier-series-8196721e7a3a)
 - [Part II](https://medium.com/@jcatankard_76170/forecasting-with-fourier-series-part-ii-f74bdeaf1722)

### Seasonality:
- For best performance, aim to include at least two cycles of the largest seasonal component to be fitted 
- Multiple waves per seasonality are fit according to the specified Fourier order
    - more terms results in better fit but can result in over-fitting
- Seasonal periods are set by pre-determined values:
   - weekly seasonality: 7 days
   - monthly seasonality: 30.43 days
   - quarterly seasonality: 91.31 days
   - yearly seasonality: 365.25 days
- If seasonal terms interact with each other (i.e. are not just additive), it may be best to log transform your time-series before fitting
  - where necessary it may also be appropriate to log transform some or all regressors

### Future updates:
 - prediction intervals
 - add autoregression lags
 - add deploy to pip into pipeline

## FourierForecast
### Parameters
- growth: str, default='linear'. Possible values 'linear', 'flat', 'logistic', 'logarithmic'.
- weekly_seasonality_terms: int, default=3
    - number of Fourier terms to generate for fitting seasonality of 7 days
- monthly_seasonality_terms: int, default=0
    - number of Fourier terms to generate for fitting seasonality of 30.43 days
- quarterly_seasonality_terms: int, default=0
    - number of Fourier terms to generate for fitting seasonality of 91.31 days
- yearly_seasonality_terms: int, default=10
    - number of Fourier terms to generate for fitting seasonality of 365.25 days
- trend_reg: float, default=0
    - parameter regulating strength of trend fit
    - smaller values (~0-1) allow the model to fit larger seasonal fluctuations,
    - larger values (~1-100) dampen the seasonality.
- seasonality_reg: float, default=0
    - parameter regulating strength of seasonality fit
    - smaller values (~0-1) allow the model to fit larger seasonal fluctuations,
    - larger values (~1-100) dampen the seasonality.
- regressor_reg: float, default=0
    - parameter regulating strength of regressors fit
    - smaller values (~0-1) allow the model to fit larger seasonal fluctuations,
    - larger values (~1-100) dampen the seasonality.
- log_y: bool, default=True
    - takes the natural logarithm of the timeseries before fitting (and the exponent after predicting)
    - all values must be positive or reverts bact to False
    - useful for fitting interactive effects between seasonality, trend and regressors

### Methods
 - fit
   - y: NDArray[float]
     - daily time-series ordered by date
   - regressors: NDArray[float], default=None
     - optional regressors for fitting non-seasonal components ordered by date
   - sample_weight: NDArray[float], default=None
     - individual weights for each sample
 - predict
   - h: int, default=1
     - number of horizons to predict
   - regressors: NDArray[float], default=None
     - regressors corresponding to days to predict 
     - if regressors are present during fitting, these must have the same number of features
     - if None is passed, then all values will assume to be 0.
 - fitted
   - returns fitted values


 - plot_components
   - plots bias, trends, seasonalities, regressors and noise
 - plot_seasonality_components
   - plots seasonalities only
 - plot_regression_components
   - regressor_names: list[str], default=None
   - plots individual regressors as their own subplot

## Examples
### fit and predict example
```python
from fourier_forecast.fourier_forecast import FourierForecast
import matplotlib.pyplot as plt


dates = ...
actuals = ...

train_test_split = .8
n_train = int(len(dates) * train_test_split)
n_predict = len(dates) - n_train

train_dates = dates[: n_train]
train_actuals = actuals[: n_train]

ff = FourierForecast()
                     
ff.fit(train_actuals)
preds = ff.predict(h=n_predict)

plt.plot(dates, actuals, label='actuals')
plt.plot(train_dates, preds[: n_train], label='train')
plt.plot(dates[n_train: ], preds, label='preds')
plt.legend()
plt.show()
```
<p float="left">
  <img src="./images/example_train_preds.png" width="100%" />
</p>

### regressor example with plot_components()
```python
from fourier_forecast.fourier_forecast import FourierForecast


actuals = ...
regressors = ...

ff = FourierForecast(weekly_seasonality_terms=1,
                     monthly_seasonality_terms=1,
                     quarterly_seasonality_terms=1,
                     yearly_seasonality_terms=1
                     )
                     
ff.fit(actuals, regressors)
ff.plot_components()
```
<p float="left">
  <img src="./images/example_plot_components.png" width="100%" />
</p>

### fourier order example with plot_components()
```python
from fourier_forecast.fourier_forecast import FourierForecast


actuals = ...

ff = FourierForecast(weekly_seasonality_terms=3,
                     monthly_seasonality_terms=0,
                     quarterly_seasonality_terms=0,
                     yearly_seasonality_terms=10
                     )      
ff.fit(actuals)
ff.plot_components()
```
<p float="left">
  <img src="./images/example_fourier_order.png" width="100%" />
</p>

### multiplicative seasonality example
```python
from fourier_forecast.fourier_forecast import FourierForecast
import numpy as np


dates = ...
actuals = ...
regressors = ...

train_test_split = .8
n_train = int(len(dates) * train_test_split)
n_predict = len(dates) - n_train

ff = FourierForecast() 
ff.fit(y=np.log(actuals[: n_train]),
       regressors=regressors[: n_train]
       )
preds = np.exp(
    ff.predict(h=n_predict, regressors=regressors[n_train: ])
)
mape = np.absolute(preds / actuals[n_train: ] - 1) / n_predict
print(mape)
```