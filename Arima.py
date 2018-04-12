from pandas import Series
from statsmodels.tsa.arima_model import ARIMA
from scipy.stats import boxcox
import numpy
from math import sqrt
from math import log
from math import exp


# monkey patch around bug in ARIMA class
def __getnewargs__(self):
    return ((self.endog), (self.k_lags, self.k_diff, self.k_ma))


ARIMA.__getnewargs__ = __getnewargs__

# load data
# series = Series.from_csv('dataset.csv', header=0)
# split_point = len(series) - 12
# dataset, validation = series[0:split_point], series[split_point:]
# print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
# dataset.to_csv('dataset1.csv')
# validation.to_csv('validation.csv')
def boxcox_inverse(value, lam):
    if lam == 0:
        return exp(value)
    return exp(log(lam * value + 1) / lam)


series = Series.from_csv('dataset1.csv')
# prepare data
X = series.values
X = X.astype('float32')
# transform data
transformed = boxcox(X)

# fit model
model = ARIMA(X, order=(0, 1, 2))
model_fit = model.fit(disp=0)