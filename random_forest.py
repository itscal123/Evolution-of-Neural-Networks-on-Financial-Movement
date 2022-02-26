import numpy as np
import random
from matplotlib import pyplot as plt
from datetime import datetime
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def getData():
    """
    Random walk model simply needs to predict the open price from previous timestep with some random noise
    params: arr (NumPy array)
    returns: output (Numpy array of size 1 x 5)
    """
    output = np.load("data\\tensor.npy")
    return output[:,562:,:]


def observed(arr):
    """
    Takes the data NumPy array and converts into an array of all observed prices.
    param: arr (NumPy array used for other model training data)
    returns: yhat (NumPy array containing all the observations)
    """
    y = None
    for t in range(arr[:,:,1].shape[1]):
        prev = arr[:,t,1]
        if y is None:
            y = prev
        y = np.column_stack((y, prev))
    return y[:,1:]


def float2date(num):
    """
    Takes the numerical representation of the date and returns it as a datetime object
    for matplotlib
    params: num (int)
    returns: datetime object
    """
    string = str(num)
    year = int(string[0:4])
    month = int(string[4:6])
    day = int(string[6:8])
    return datetime(year, month, day)


def residuals(y, yhat):
    """
    Calculates the residuals between the observed price and the predicted price
    params: y (NumPy array of observed prices), yhat (NumPy array of predicted prices)
    returns: residuals (NumPy array of the differences between y and yhat)
    """
    return [pair[0] - pair[1] for pair in zip(y, yhat)]


def createPlots(y, yhat):
    """
    Function that creates plots of the predicted, observed, and residual values of the 
    portfolio.
    params: None
    returns: None 
    """
    data = getData()
    x = [float2date(day) for day in data[0,:,0]]

    # Apple
    fig, axs = plt.subplots(2, figsize=(15,8))
    axs[0].plot(x, yhat, label="Predictions")
    axs[0].plot(x, y, label="Observed", color="darkorange")
    axs[0].legend(loc=2)
    axs[0].title.set_text("Predicted vs Observed Open Prices for Apple")

    e = residuals(y, yhat)
    axs[1].plot(x, e, color="g", label="Residuals")
    axs[1].legend(loc=2)
    axs[1].title.set_text("Residuals for Apple's Predictions")
    
    # Amazon
    fig, axs = plt.subplots(2, figsize=(15,8))
    axs[0].plot(x, yhat[1], label="Predictions")
    axs[0].plot(x, y[1], label="Observed", color="darkorange")
    axs[0].legend(loc=2)
    axs[0].title.set_text("Predicted vs Observed Open Prices for Amazon")

    e = residuals(y[1], yhat[1])
    axs[1].plot(x, e, color="g", label="Residuals")
    axs[1].legend(loc=2)
    axs[1].title.set_text("Residuals for Amazon's Predictions")

    # Microsoft
    fig, axs = plt.subplots(2, figsize=(15,8))
    axs[0].plot(x, yhat[2], label="Predictions")
    axs[0].plot(x, y[2], label="Observed", color="darkorange")
    axs[0].legend(loc=2)
    axs[0].title.set_text("Predicted vs Observed Open Prices for Microsoft")

    e = residuals(y[2], yhat[2])
    axs[1].plot(x, e, color="g", label="Residuals")
    axs[1].legend(loc=2)
    axs[1].title.set_text("Residuals for Microsoft's Predictions")
    plt.show()


def movement(y, yhat):
    """
    Generates the total gains/loss on the portfolio following a simple buy low sell high
    policy on the predicted stock prices. 
    params: pred (Numpy array of predicted prices), obs (Numpy array of observed prices)
    returns: portfolio value (float)
    """
    correct, incorrect = 0, 0
    for i in range(len(yhat)):
        if yhat[i] < yhat[i-1]:
            if y[i] < y[i-1]:
                correct += 1
            else:
                incorrect += 1
        elif y[i] > y[i-1]:
            if y[i] > y[i-1]:
                correct += 1
            else:
                incorrect += 1
        else:
            incorrect += 1
    return correct/(incorrect+correct)


def createModel(X, y):
    model = RandomForestRegressor(n_jobs=-1)
    model.fit(X, y)
    return model


def preprocessData(data):
    X, Y = offset(data)
    return trainTestSplit(X, Y)


def trainTestSplit(X, Y):
    X_train, X_test = X[:-20,:], X[-20:,:]
    Y_train, Y_test = Y[:-20], Y[-20:]
    return X_train, X_test, Y_train, Y_test


def offset(data):
    X = data[:-1][::-1]
    Y = data[1:,1][::-1]
    return X, Y


if __name__ == "__main__":
    data = getData()
    stacked_data = []
    apple_data, amazon_data, microsoft_data = data[0], data[1], data[2]
    f = open("experiments\\random_forest.txt", "w")
    for _ in range(30):
        accuracy = []
        for dataset in [apple_data, amazon_data, microsoft_data]:
            X_train, X_test, y_train, y_test = preprocessData(dataset)
            model = createModel(X_train, y_train)
            yhat = model.predict(X_train)

            data = getData()
            x = [float2date(day) for day in data[0,:,0]][::-1]

            # Test data
            yhat = model.predict(X_test)
            """
            fig, axs = plt.subplots(2, figsize=(15,8))
            axs[0].plot(x[-20:], yhat, label="Predictions")
            axs[0].plot(x[-20:], y_test, label="Observed", color="darkorange")
            axs[0].legend(loc=2)
            axs[0].title.set_text("Predicted vs Observed Open Prices for Apple")

            e = residuals(y_test, yhat)
            axs[1].plot(x[-20:], e, color="g", label="Residuals")
            axs[1].legend(loc=2)
            axs[1].title.set_text("Residuals for Apple's Predictions")
            accuracy.append(movement(y_test, yhat))
            """
            accuracy.append(movement(y_test, yhat))
        f.write(str(sum(accuracy) / 3) + ",")
        

    df = pd.read_csv("experiments\\random_forest.txt", header=None).T
    print(df.describe())
