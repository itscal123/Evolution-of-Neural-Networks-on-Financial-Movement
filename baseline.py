import numpy as np
import random
from matplotlib import pyplot as plt
from datetime import datetime
import pandas as pd


def getData():
    """
    Random walk model simply needs to predict the open price from previous timestep with some random noise
    params: arr (NumPy array)
    returns: output (Numpy array of size 1 x 5)
    """
    output = np.load("data\\tensor.npy")
    return output[:,562:,:]


def predict(arr):
    """
    Takes NumPy array of stock prices, then returns a NumPy array of the same size containing
    the predictions for the next time step (with some random noise)
    param: arr (NumPy array of size 3x3)
    returns: output (Numpy array of size 3x3)
    """
    for i in range(3):
        mu = arr[i]
        sigma = 0.1 * mu
        arr[i] = random.gauss(mu, sigma)

    return arr


def predictions(arr):
    """
    Takes a NumPy array of stock price data and returns an array of all predictions. 
    param: arr (NumPy array used for other model training data)
    returns: yhat (Numpy array containing all the predictions)
    """
    yhat = None
    for t in range(arr[:,:,1].shape[1]):
        prev = arr[:,t,1]
        future = predict(prev)
        if yhat is None:
            yhat = prev
        yhat = np.column_stack((yhat, future))
    return yhat[:,1:]


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


def createPlots():
    """
    Function that creates plots of the predicted, observed, and residual values of the 
    portfolio.
    params: None
    returns: None 
    """
    data = getData()
    x = [float2date(day) for day in data[0,::-1,0]]
    yhat = predictions(getData())[:, ::-1]
    y = observed(getData())[:, ::-1]
    split = x.index(datetime(2014, 7, 8))
    last = x.index(datetime(2014, 6, 9))
    # Apple
    for i in range(len(x)):
        if i >= last and i < split:
            y[0][i] += 500
            yhat[0][i] += 500
        if i >= split:
            y[0][i] *= 7
            yhat[0][i] *= 7
    fig, axs = plt.subplots(2, figsize=(15,8))
    axs[0].plot(x, yhat[0][::], label="Predictions")
    axs[0].plot(x, y[0][::], label="Observed", color="darkorange")
    axs[0].legend(loc=2)
    axs[0].title.set_text("Predicted vs Observed Open Split Adjusted Prices for Apple (Random Walk)")
    axs[0].set_ylabel("Price (USD)")
    axs[0].grid(True)

    e = residuals(y[0], yhat[0])
    axs[1].plot(x, e, color="g", label="Residuals")
    axs[1].legend(loc=2)
    axs[1].title.set_text("Residuals for Apple's Predictions (Random Walk)")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Price (USD)")
    axs[1].grid(True)
    
    # Amazon
    fig, axs = plt.subplots(2, figsize=(15,8))
    axs[0].plot(x, yhat[1], label="Predictions")
    axs[0].plot(x, y[1], label="Observed", color="darkorange")
    axs[0].legend(loc=2)
    axs[0].title.set_text("Predicted vs Observed Open Prices for Amazon (Random Walk)")
    axs[0].set_ylabel("Price (USD)")
    axs[0].grid(True)

    e = residuals(y[1], yhat[1])
    axs[1].plot(x, e, color="g", label="Residuals")
    axs[1].legend(loc=2)
    axs[1].title.set_text("Residuals for Amazon's Predictions (Random Walk)")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Price (USD)")
    axs[1].grid(True)

    # Microsoft
    fig, axs = plt.subplots(2, figsize=(15,8))
    axs[0].plot(x, yhat[2], label="Predictions")
    axs[0].plot(x, y[2], label="Observed", color="darkorange")
    axs[0].legend(loc=2)
    axs[0].title.set_text("Predicted vs Observed Open Prices for Microsoft (Random Walk)")
    axs[0].set_ylabel("Price (USD)")
    axs[0].grid(True)

    e = residuals(y[2], yhat[2])
    axs[1].plot(x, e, color="g", label="Residuals")
    axs[1].legend(loc=2)
    axs[1].title.set_text("Residuals for Microsoft's Predictions (Random Walk)")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Price (USD)")
    axs[1].grid(True)
    plt.show()


def movement():
    """
    Generates the total gains/loss on the portfolio following a simple buy low sell high
    policy on the predicted stock prices. 
    params: pred (Numpy array of predicted prices), obs (Numpy array of observed prices)
    returns: portfolio value (float)
    """
    yhat = predictions(getData())
    y = observed(getData())
    correct, incorrect = 0, 0
    for stock in range(yhat.shape[0]):
        for i in range(1, yhat.shape[1]):
                if yhat[stock][i] < yhat[stock][i-1]:
                    if y[stock][i] < y[stock][i-1]:
                        correct += 1
                    else:
                        incorrect += 1
                elif y[stock][i] > y[stock][i-1]:
                    if y[stock][i] > y[stock][i-1]:
                        correct += 1
                    else:
                        incorrect += 1
                else:
                    incorrect += 1
    return correct/(incorrect+correct)

def generateData(n=40):
    """
    Generates the data that will be used in the two sample z-test later. 
    params: None
    returns: None
    """
    print("Generating Data points...")
    f = open("experiments\\baseline.txt", "w")
    for i in range(n):
        output = f'{str(movement())},'
        f.write(output)
    print("Complete!")


def summaryStats():
    """
    Outputs summary statistics for the experimental data
    params: None
    returns: None
    """
    df = pd.read_csv("experiments\\baseline.txt", header=None).T
    print(df.describe())


def mae():
    yhat = predictions(getData())
    y = observed(getData())

    apple = [abs(i) for i in residuals(y[0], yhat[0])]
    amazon = [abs(i) for i in residuals(y[1], yhat[1])]
    microsoft = [abs(i) for i in residuals(y[2], yhat[2])]

    apple = sum(apple) / len(apple)
    amazon = sum(amazon) / len(amazon)
    microsoft = sum(microsoft) / len(microsoft)

    return apple + amazon + microsoft

if __name__ == "__main__":
    createPlots()
    #generateData()
    #summaryStats()
    #print(mae())