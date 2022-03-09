import datetime as dt
import pandas_datareader.data as web
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model

def getData():
    output = np.load("data\\tensor.npy")
    return output[:,562:,:]


def returns():
    data = getData()
    apple_adj = web.DataReader('AAPL', 'yahoo', start='1999-11-26', end='2019-10-23')["Adj Close"]
    amzn_adj = web.DataReader('AMZN', 'yahoo', start='1999-11-26', end='2019-10-23')["Adj Close"]
    msft_adj = web.DataReader('MSFT', 'yahoo', start='1999-11-26', end='2019-10-23')["Adj Close"]
    
    apple_returns = 100 * apple_adj.pct_change().dropna()
    amzn_returns = 100 * amzn_adj.pct_change().dropna()
    msft_returns = 100 * msft_adj.pct_change().dropna()

    apple_new = np.insert(data[0], 1, apple_returns.to_numpy(), axis=1)
    amzn_new = np.insert(data[1], 1, amzn_returns.to_numpy(), axis=1)
    msft_new = np.insert(data[2], 1, msft_returns.to_numpy(), axis=1)
    new_data = np.stack((apple_new, amzn_new, msft_new), axis=0)
    return new_data


def getTrainTest(stock):
    """
    param: stock is a the symbol for either AAPL, AMZN, or MSFT
    """
    data = returns()
    X = data[:,1:,:,]
    Y = data[:,:-1,1]
    X = np.flip(X, 1)
    Y = np.flip(Y, 1)

    X_train, X_test = X[:,:-20,:], X[:,-20:,:]
    Y_train, Y_test = Y[:,:-20], Y[:,-20:]

    if stock == "AAPL":
        i = 0
    elif stock == "AMZN":
        i = 1
    else:
        i = 2
    
    return X_train[i], X_test[i], Y_train[i], Y_test[i]


def garchModel(y_train, y_test):
    # define model
    model = arch_model(np.ascontiguousarray(y_train), mean="Zero", vol="GARCH", p=15, q=15)

    # fit model
    model_fit = model.fit()

    # forecast test set
    yhat = model_fit.forecast(horizon=20, reindex=True)

    # plot forecast variance
    plt.plot(yhat.variance.values[-1, :])
 

    plt.show()


def variance(y_test):
    for i in range(len(y_test)-1):
        r_bar = (y_test[i] + y_test[i+1]) / 2
        r = y_test[i+1]
if __name__ == "__main__":
    X_train, X_test, Y_train, Y_test = getTrainTest("AAPL")
    garchModel(Y_train, Y_test)


