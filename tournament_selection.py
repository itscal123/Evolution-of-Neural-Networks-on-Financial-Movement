import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow_addons as tfa
from sklearn.preprocessing import MinMaxScaler
from random import randrange
import copy
import random
import string
import pandas as pd


def window(X, Y, start=0, end=5):
    """
    Sliding window function to splice data into batches of size end - start for
    the model to use past data in its training and predictions.
    params:
        X: NumPy array of the raw feature values
        Y: Numpy array of all the target values
        start: Start of the window
        end: End of the window
    returns:
        X_new: Correctly spliced version of X
        Y_new: Correctly spliced version of Y
    """
    X_new = []
    Y_new = []
    window = end
    for i in range(X.shape[1]-end):
        X_new.append(X[:,start:end,:])
        start += 1
        end += 1
    
    for i in range(Y.shape[1]-window):
        Y_new.append(Y[:,i+window])

    X_new = np.asarray(X_new)
    Y_new = np.asarray(Y_new)
    return X_new, Y_new


def prepareData():
    """
    Splits the data into training and test sets. Also applies min max scaling and the
    window function to process all data to be fed directly into training and inference.
    """
    # Split Data into training/test
    raw_data = getData()
    X = raw_data[:,1:,:,]
    Y = raw_data[:,:-1,1]
    X = np.flip(X, 1)
    Y = np.flip(Y, 1)

    X_train, X_test = X[:,:-20,:], X[:,-20:,:]
    Y_train, Y_test = Y[:,:-20], Y[:,-20:]

    # Apply MinMaxScaler to each stock separately in the training data
    appleScaler, amznScaler, msftScaler = MinMaxScaler(), MinMaxScaler(), MinMaxScaler()
    appleTrain = appleScaler.fit_transform(X_train[0])
    amznTrain = amznScaler.fit_transform(X_train[1])
    msftTrain = msftScaler.fit_transform(X_train[2])
    X_train = np.stack([appleTrain, amznTrain, msftTrain])

    # Apply MinMaxScaler transform on the test data
    appleTest = appleScaler.transform(X_test[0])
    amznTest = amznScaler.transform(X_test[1])
    msftTest = msftScaler.transform(X_test[2])
    X_test = np.stack([appleTest, amznTest, msftTest])

    X_train, Y_train = window(X_train, Y_train)
    X_test, Y_test = window(X_test, Y_test)
    return X_train, X_test, Y_train, Y_test

def getData():
    """
    Random walk model simply needs to predict the open price from previous timestep with some random noise
    params: arr (NumPy array)
    returns: output (Numpy array of size 1 x 5)
    """
    output = np.load("data\\tensor.npy")[:,562:,:]
    return output

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
    return [pair[0] - pair[1] for pair in zip(y, yhat)]


def createPlots(model, X_train, Y_train):
    data = getData()
    y = Y_train.T
    yhat = model.predict(X_train).T
    x = [float2date(day) for day in data[0,::-1,0]][26:]
    # Apple
    fig, axs = plt.subplots(2, figsize=(15,8))
    axs[0].plot(x, yhat[0][::], label="Predictions")
    axs[0].plot(x, y[0][::], label="Observed", color="darkorange")
    axs[0].legend(loc=2)
    axs[0].title.set_text("Predicted vs Observed Open Prices for Apple (Tournament Selection)")
    axs[0].set_ylabel("Price (USD)")
    axs[0].grid(True)

    e = residuals(y[0][::], yhat[0][::])
    axs[1].plot(x, e, color="g", label="Residuals")
    axs[1].legend(loc=2)
    axs[1].title.set_text("Residuals for Apple's Predictions (Tournament Selection)")
    axs[1].set_ylabel("Price (USD)")
    axs[1].set_xlabel("Years")
    axs[1].grid(True)

    # Amazon
    fig, axs = plt.subplots(2, figsize=(15,8))
    axs[0].plot(x, yhat[1][::], label="Predictions")
    axs[0].plot(x, y[1][::], label="Observed", color="darkorange")
    axs[0].legend(loc=2)
    axs[0].title.set_text("Predicted vs Observed Open Prices for Amazon (Tournament Selection)")
    axs[0].set_ylabel("Price (USD)")
    axs[0].grid(True)

    e = residuals(y[1][::], yhat[1][::])
    axs[1].plot(x, e, color="g", label="Residuals")
    axs[1].legend(loc=2)
    axs[1].title.set_text("Residuals for Amazon's Predictions (Tournament Selection)")
    axs[1].set_ylabel("Price (USD)")
    axs[1].set_xlabel("Years")
    axs[1].grid(True)

    # Microsoft
    fig, axs = plt.subplots(2, figsize=(15,8))
    axs[0].plot(x, yhat[2][::], label="Predictions")
    axs[0].plot(x, y[2][::], label="Observed", color="darkorange")
    axs[0].legend(loc=2)
    axs[0].title.set_text("Predicted vs Observed Open Prices for Microsoft (Tournament Selection)")
    axs[0].set_ylabel("Price (USD)")
    axs[0].grid(True)

    e = residuals(y[2][::], yhat[2][::])
    axs[1].plot(x, e, color="g", label="Residuals")
    axs[1].legend(loc=2)
    axs[1].title.set_text("Residuals for Microsoft's Predictions (Tournament Selection)")
    axs[1].set_ylabel("Price (USD)")
    axs[1].set_xlabel("Years")
    axs[1].grid(True)    
    plt.show()
    return


def createValPlots(model, X, Y, offset=15):
    Y = Y.T
    Yhat = model.predict(X).T
    data = getData()
    x = [float2date(day) for day in data[0,::-1,0]][-offset:]
    # Apple
    fig, axs = plt.subplots(2, figsize=(15,8))
    axs[0].plot(x, Yhat[0][::], label="Predictions")
    axs[0].plot(x, Y [0][::], label="Observed", color="darkorange")
    axs[0].legend(loc=2)
    axs[0].title.set_text("Predicted vs Observed Open Prices for Apple (Tournament Selection)")
    axs[0].grid(True)
    axs[0].set_ylabel("Price (USD)")

    e = residuals(Y[0][::], Yhat[0][::])
    axs[1].plot(x, e, color="g", label="Residuals")
    axs[1].legend(loc=2)
    axs[1].title.set_text("Residuals for Apple's Predictions (Tournament Selection)")
    axs[1].set_ylabel("Price (USD)")
    axs[1].set_xlabel("Days")
    axs[1].grid(True)

    # Amazon
    fig, axs = plt.subplots(2, figsize=(15,8))
    axs[0].plot(x, Yhat[1][::], label="Predictions")
    axs[0].plot(x, Y[1][::], label="Observed", color="darkorange")
    axs[0].legend(loc=2)
    axs[0].title.set_text("Predicted vs Observed Open Prices for Amazon (Tournament Selection)")
    axs[0].set_ylabel("Price (USD)")
    axs[0].grid(True)

    e = residuals(Y[1][::], Yhat[1][::])
    axs[1].plot(x, e, color="g", label="Residuals")
    axs[1].legend(loc=2)
    axs[1].title.set_text("Residuals for Amazon's Predictions (Tournament Selection)")
    axs[1].set_ylabel("Price (USD)")
    axs[1].set_xlabel("Days")
    axs[1].grid(True)

    # Microsoft
    fig, axs = plt.subplots(2, figsize=(15,8))
    axs[0].plot(x, Yhat[2][::], label="Predictions")
    axs[0].plot(x, Y[2][::], label="Observed", color="darkorange")
    axs[0].legend(loc=2)
    axs[0].title.set_text("Predicted vs Observed Open Prices for Microsoft (Tournament Selection)")
    axs[0].set_ylabel("Price (USD)")
    axs[0].grid(True)

    e = residuals(Y[2][::], Yhat[2][::])
    axs[1].plot(x, e, color="g", label="Residuals")
    axs[1].legend(loc=2)
    axs[1].title.set_text("Residuals for Microsoft's Predictions (Tournament Selection)")
    axs[1].set_ylabel("Price (USD)")
    axs[1].set_xlabel("Days")
    axs[1].grid(True)
    plt.show()
    return


def alterNames(config):
    for layer in config["layers"]:
        layer["config"]["name"] = ''.join(random.choice(string.ascii_lowercase) for i in range(10))
        if "cell" in layer["config"]:
            layer["config"]["cell"]["config"]["name"] = ''.join(random.choice(string.ascii_lowercase) for i in range(10))
    return config
    

def alterLearningRate(parent):
    child_config = parent.get_config()
    child_config = alterNames(child_config)
    child = keras.Sequential.from_config(child_config)
    learning_rate = randrange(1, 100) / 10000
    childOptimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    child.compile(optimizer=childOptimizer, loss="mse")
    return child


def alterStride(parent):
    child_config = parent.get_config()
    convolutions = []

    for i in range(len(child_config["layers"])):
        if child_config["layers"][i]["class_name"] == "Conv1D":
            convolutions.append(i) 

    layer = randrange(convolutions[0], convolutions[-1]+1)
    delta = 1 if randrange(0,2) == 1 else -1
    stride = child_config["layers"][layer]["config"]["strides"]
    if stride[0] == 2:
        child_config["layers"][layer]["config"]["strides"] = (stride[0] + 1, )
    else:
        child_config["layers"][layer]["config"]["strides"] = (stride[0] + delta, )
    
    child_config = alterNames(child_config)
    child = keras.Sequential.from_config(child_config)
    return child  


def insert1dConv(parent):
    child_config = parent.get_config()
    convolutions = []

    for i in range(len(child_config["layers"])):
        if child_config["layers"][i]["class_name"] == "Conv1D":
            convolutions.append(i)

    layer = randrange(convolutions[0], convolutions[-1]+1)

    if layer == 1:
        new = copy.deepcopy(child_config["layers"][layer])
        new["config"]["filters"] = 20
        new["config"]["padding"] = "causal"
    else:
        new = copy.deepcopy(child_config["layers"][layer])
        if new["config"]["filters"] == 1:
            child_config["layers"].insert(layer+1, new)
        else:
            new["config"]["filters"] = new["config"]["filters"] // 2

    child_config["layers"].insert(layer+1, new)
    child_config = alterNames(child_config)
    child = keras.Sequential.from_config(child_config)
    return child


def remove1dConv(parent):
    child_config = parent.get_config()
    convolutions = []

    for i in range(len(child_config["layers"])):
        if child_config["layers"][i]["class_name"] == "Conv1D":
            convolutions.append(i)

    layer = randrange(convolutions[0], convolutions[-1]+1)

    if layer == 1:
        return keras.models.clone_model(parent)
    else:
        if child_config["layers"][layer+1]["class_name"] == "Reshape":
            prev = child_config["layers"][layer-1]["config"]["filters"]
            shape = (3, prev)
            child_config["layers"][layer+1]["config"]["target_shape"] = shape
        child_config["layers"].pop(layer)
        child_config = alterNames(child_config)
        child = keras.Sequential.from_config(child_config)
        return child

    
def insertPeepLSTM(parent):
    child_config = parent.get_config()
    lstms = []

    for i in range(len(child_config["layers"])):
        if child_config["layers"][i]["class_name"] == "RNN":
            lstms.append(i)
    
    while True:
        layer = randrange(lstms[0], lstms[-1]+1)
        if layer in lstms:
            break
        
    new = copy.deepcopy(child_config["layers"][layer])
    dropout = {'class_name': 'Dropout',
                'config': {'name': 'placeholder',
                            'trainable': True,
                            'dtype': 'float32',
                            'rate': 0.25,
                            'noise_shape': None,
                            'seed': None}}
    

    if layer == lstms[0]:
        if child_config["layers"][layer+1]["class_name"] == "Dense":
            new["config"]["return_sequences"] = False
        else:
            new["config"]["return_sequences"] = True
    elif layer == lstms[-1]:
        child_config["layers"][layer]["config"]["return_sequences"] = True
        new["config"]["return_sequences"] = False
    else:
        new["config"]["return_sequences"] = True
    
    child_config["layers"].insert(layer+1, new)
    child_config["layers"].insert(layer+2, dropout)
    child_config = alterNames(child_config)
    child = keras.Sequential.from_config(child_config)
    return child


def removePeepLSTM(parent):
    child_config = parent.get_config()
    lstms = []

    for i in range(len(child_config["layers"])):
        if child_config["layers"][i]["class_name"] == "RNN":
            lstms.append(i) 

    while True:
        layer = randrange(lstms[0], lstms[-1]+1)
        if layer in lstms:
            break

    if layer == lstms[0]:
        pass
    elif layer == lstms[-1]:
        child_config["layers"][layer-1]["config"]["return_sequences"] = False
        child_config["layers"].pop(layer+1)
        child_config["layers"].pop(layer)
    else:
        child_config["layers"].pop(layer+1)
        child_config["layers"].pop(layer)

    child_config = alterNames(child_config)
    child = keras.Sequential.from_config(child_config)
    return child


def mutate(parent):
    """
    Mutated version of the model (parent)
    """
    mutation = randrange(7)
    # Do nothing
    if mutation == 0:
        child = keras.models.clone_model(parent)
        return child
    # Alter Learning Rate
    elif mutation == 1:
        return alterLearningRate(parent)
    # Alter Stride
    elif mutation == 2:
        return alterStride(parent)
    # Insert 1D Convolution
    elif mutation == 3:
        return insert1dConv(parent)
    # Remove 1D Convolution
    elif mutation == 4:
        return remove1dConv(parent)
    # Insert Peephole LSTM
    elif mutation == 5:
        return insertPeepLSTM(parent)
    # Remove Peephole LSTM
    else:
        return removePeepLSTM(parent)


def tournament_selection(model, X_train, Y_train):
    generation = 1
    population = []
    while(generation != 5):
        model = mutate(model)
        model.compile(optimizer="rmsprop", loss="mse")
        history = model.fit(X_train, Y_train, epochs=20, batch_size=128, shuffle=False)
        population.append([model, history, history.history["loss"][-1]])
        generation += 1
    return population


def movement(model, X, Y):
    """
    Generates the total gains/loss on the portfolio following a simple buy low sell high
    policy on the predicted stock prices. 
    params: pred (Numpy array of predicted prices), obs (Numpy array of observed prices)
    returns: portfolio value (float)
    """
    Y = Y.T
    yhat = model.predict(X_train).T
    correct, incorrect = 0, 0
    for stock in range(yhat.shape[0]):
        for i in range(1, yhat.shape[1]):
            if yhat[stock][i] <= yhat[stock][i-1]:
                if Y[stock][i] <= Y[stock][i-1]:
                    correct += 1
                else:
                    incorrect += 1
            else:
                if Y[stock][i] > Y[stock][i-1]:
                    correct += 1
                else:
                    incorrect += 1
        return correct/(incorrect+correct)


def generateData(n=10):
    print("Generating Data points...")
    # Base
    LSTMCell = tfa.rnn.PeepholeLSTMCell(32)
    LSTMCell2 = tfa.rnn.PeepholeLSTMCell(32)

    base = keras.Sequential([
    keras.layers.Conv1D(filters=20, kernel_size=4, strides=2, padding="valid", input_shape=[3,5,13]),
    keras.layers.Reshape((3,20)),
    keras.layers.RNN(LSTMCell, return_sequences=True),
    keras.layers.RNN(LSTMCell2),
    keras.layers.Dense(3)
    ])
    optimizer = keras.optimizers.RMSprop(learning_rate=1e-3)
    base.compile(optimizer=optimizer, loss="mse")

    f = open("experiments\\tournament.txt", "a")
    success, fail = 0, 0
    for i in range(n):
        print("Iteration {} of Tournament Selection".format(i+1))
        try:
            population = tournament_selection(base, X_train, Y_train)
            best = min(population, key=lambda x: x[-1])[0]
            callback = keras.callbacks.EarlyStopping(monitor="loss", min_delta=50, patience=8, restore_best_weights=True)
            history = best.fit(X_train, Y_train, epochs=5000, batch_size=128, callbacks=[callback], shuffle=False)
            output = f'{str(movement(best, X_train, Y_train))},'
            f.write(output)
            success += 1
            best.save("models/iteration_{}".format(i+50))
        except Exception as e:
            print("There was an error at iteration {}".format(i))
            print(e)
            fail += 1
    print("Report")
    print(f'Success: {success}\t Fail:{fail}')
    print("Complete!")


def summaryStats():
    """
    Outputs summary statistics for the experimental data
    params: None
    returns: None
    """
    df = pd.read_csv("experiments\\tournament_final.txt", header=None).T
    print(df.describe())


if __name__ == "__main__":
    # Get the train/test data
    X_train, X_test, Y_train, Y_test = prepareData()

    # Conduct tournament selection
    #generateData()

    #model = keras.models.load_model("models\iteration_5")

    # Retrain the best model
    #callback = keras.callbacks.EarlyStopping(monitor="loss", min_delta=15, patience=8, restore_best_weights=True)
    #history = model.fit(X_train, Y_train, epochs=25, batch_size=8, callbacks=[callback], shuffle=False)

    # Save the final model
    #model.save("models\\best")

    # Load the final model
    model = keras.models.load_model("models\\best")

    # Model summary
    #model.summary

    # Create plots
    createPlots(model, X_train, Y_train)
    createValPlots(model, X_test, Y_test)
    #summaryStats()