# FourierForecast
A time-series model that decomposes a daily time-series into seasonality, trend, bias, optional regressors and noise.
 
- The initial seasonality amplitude and phase are inferred by Fast Fourier Transformation
- This is then fine-tuned and fitted with the other components using gradient descent
- Components can be visualised with plot_components method
- Numba is used for performance boost

The main purpose of this is to educate myself on Fast Fourier Transformations and gradient descent.
It is not advised to use this at this point in time.

### Seasonality:
- In theory seasonal components can be fitted with less training data than a full season
  - e.g. yearly seasonality can be fitted even when the training data covers just 9 months
  - however, for best performance, aim to include at least two cycles of the largest seasonal component to be fitted
  - at this moment, only one wave is fitted per seasonal component which means nuance is lost at a granular level
  - fitting frequency by gradient descent does not result in stable outcomes so only amplitude and phase are fitted
  whilst frequency is determined by pre-determined values:
     - weekly_seasonality: 7 days
     - monthly_seasonality: 30.43 days
     - quarterly_seasonality: 91.31 days
     - yearly_seasonality: 365.25 days
  - For multiplicative seasonality, it may be best to log transform your time-series before fitting
    - where necessary it may also be appropriate to log transform some or all regressors

### Future updates:
 - weighted fitting: fit the data whilst placing greater emphasis on more recent data or periods of higher importance
 - higher fourier orders: fit multiple waves per seasonal component to create a more nuanced fit
 - confidence intervals
 - add deploy to pip into pipeline

## FourierForecast
### Parameters
 - weekly_seasonality: bool, default=True
 - monthly_seasonality: bool, default=False
 - quarterly_seasonality: bool, default=False
 - yearly_seasonality: bool, default=True
 - learning_rate: float, default=0.001
 - n_iterations: int, default=100_000
 - tol: float, default=1e-5
   - stops training if the max, absolute parameter update is smaller than tol

### Methods
 - fit
   - ds: NDArray[date] | list[date]
     - ordered dates corresponding to each row of the training data
   - y: NDArray[float]
     - daily time-series ordered by date
   - regressors: NDArray[float], default=None
     - optional regressors for fitting non-seasonal components ordered by date
 - predict
   - ds: NDArray[date] | list[date]
     - dates corresponding to days to predict
   - regressors: NDArray[float], default=None
     - regressors corresponding to days to predict 
     - if regressors are present during fitting, these must have the same number of features
     - if None is passed, then all values will assume to be 0.
 - plot_components
   - plots bias, trends, seasonality, regressors and noise

## Examples
### fit and predict example
```python
from fourier_forecast.fourier_forecast import FourierForecast
import matplotlib.pyplot as plt


dates = ...
actuals = ...

train_test_split = .8
n_train = int(len(dates) * train_test_split)

train_dates = dates[: n_train]
train_actuals = actuals[: n_train]

ff = FourierForecast()
                     
ff.fit(train_dates, train_actuals, regressors=None)
preds = ff.predict(ds=dates, regressors=None)

plt.plot(dates, actuals, label='actuals')
plt.plot(train_dates, preds[: n_train], label='train')
plt.plot(dates[n_train: ], preds[n_train: ], label='preds')
plt.legend()
plt.show()
```
<p float="left">
  <img src="./images/example_train_preds.png" width="100%" />
</p>

### regressor example with plot_components()
```python
from fourier_forecast.fourier_forecast import FourierForecast


dates = ...
actuals = ...
regressors = ...

ff = FourierForecast(monthly_seasonality=True,
                     quarterly_seasonality=True
                     )
                     
ff.fit(dates, actuals, regressors)
ff.plot_components()
```
<p float="left">
  <img src="./images/example_plot_components.png" width="100%" />
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

ff = FourierForecast() 
ff.fit(ds=dates[: n_train],
       y=np.log(actuals[: n_train]),
       regressors=regressors[: n_train]
       )
preds = np.exp(
    ff.predict(ds=dates[n_train: ],
               regressors=regressors[n_train: ]
               )
)
mape = np.absolute(preds / actuals[n_train: ] - 1) / preds.size
print(mape)
```