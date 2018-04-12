import pandas as pd
import numpy as np
import os
import c3pyo as c3
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from pandas import datetime
from pandas import Series
from pandas.core import datetools
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error


class weatherForcasting:
    def __init__(self,path,filename):
        self.__data = pd.read_csv(path+filename )
    def data_preperation(self):
        print(self.__data.head())

        self.__data = self.__data.drop(['LAT','LON','Unnamed: 6','Unnamed: 7','Unnamed: 8','Unnamed: 9','Unnamed: 11'],axis=1)
        self.__data = self.__data.rename(columns = {'ALLSKY_SFC_SW_DWN': 'SolarIrradiation'})

        self.__data['Time']= self.__data[['YEAR', 'MO','DY']].apply(lambda x : '{}-{}-{}'.format(x[0],x[1],x[2]), axis=1)
        # self.__data = self.__data.drop(['YEAR','MO','DY'], axis = 1)
        self.__data = self.__data[['Time',"SolarIrradiation",'T2M','WS50M',"DY"]]
        solardata = self.__data[['Time',"SolarIrradiation"]]
        tempdata = self.__data[['Time','T2M']]
        winddata = self.__data[['Time','WS50M']]
        winddata.to_csv('wind.csv',index=False)


        print(winddata.head(5))
        return winddata

    def data_visulazation(self):
        data = self.data_preperation()
        data.plot()

        plt.show()

    def prediction_model(self):
        data = self.data_preperation()
        data.index = pd.to_datetime(data['Time'])
        data = data.drop(['Time'], axis=1)


        def __getnewargs__(self):
            return ((self.endog), (self.k_lags, self.k_diff, self.k_ma))

        ARIMA.__getnewargs__ = __getnewargs__

        # data = Series.from_csv('wind.csv', header=0)
        print(data.head())

        X = data.values
        X = X[:-1]
        X = X.astype('float64')

        wind_tarin, wind_test = X[0:-31], X[-31:]
        # print(wind_test)
        history = [x for x in wind_tarin]
        predictions = list()
        for t in range(len(wind_test)):
            model = ARIMA(history, order=(5, 1, 0))
            model_fit = model.fit(disp=0)
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = wind_tarin[t]
            history.append(obs)
            print('predicted=%f, expected=%f' % (yhat, obs))
        error = mean_squared_error(wind_test, predictions)
        print('Test MSE: %.3f' % error)
        # plot
        pyplot.plot(wind_test)
        pyplot.plot(predictions, color='red')
        pyplot.show()


        # prediction_model = ARIMA(X,order=(5,1,0))
        # prediction_model_fit = prediction_model.fit(disp = 0)
        # print(prediction_model_fit.summary())


if __name__ == '__main__':
    filename="/La Tabatiere_temp _iradiation.csv"
    DataPath = os.path.join("dataset")
    print(DataPath)
    forcasting = weatherForcasting(DataPath,filename)
    forcasting.prediction_model()
    # forcasting.data_visulazation()