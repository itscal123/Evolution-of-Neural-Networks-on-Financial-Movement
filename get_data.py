import requests
import pandas as pd
import time
import functools
import numpy as np

#    Note that the apikey parameter in the url string should be replaced with your own api key which can be obtained for free
#    at https://www.alphavantage.co/support/

# Time Series Data
def time_series_daily(apikey, symbol):
    """
    Gets the the daily historical data for given symbol
    params: apikey (str), symbol (str)
    returns: None
    """
    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={}&outputsize=full&datatype=csv&apikey={}'.format(symbol, apikey)
    req = requests.get(url)
    content = req.content
    f = open('data\daily_{}.csv'.format(symbol), 'wb')
    f.write(content)
    f.close()
    return

def time_series_weekly(apikey, symbol):
    """
    Gets the weekly adjusted historical data for given symbol
    params: apikey (str), symbol (str)
    returns: None
    """
    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY_ADJUSTED&symbol={}&datatype=csv&apikey={}'.format(symbol, apikey)
    req = requests.get(url)
    content = req.content
    f = open('data\weekly_{}.csv'.format(symbol), 'wb')
    f.write(content)
    f.close()
    return

def time_series_monthly(apikey, symbol):
    """
    Gets the monthly adjusted historical data for given symbol
    params: apikey (str), symbol (str)
    returns: None
    """
    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY_ADJUSTED&symbol={}&datatype=csv&apikey={}'.format(symbol, apikey)
    req = requests.get(url)
    content = req.content
    f = open('data\monthly_{}.csv'.format(symbol), 'wb')
    f.write(content)
    f.close()
    return


# Economic Indicators
def real_gdp(apikey):
    """
    Gets the the quarterly real gdp
    params: apikey (str)
    returns: None
    """
    url = "https://www.alphavantage.co/query?function=REAL_GDP&interval=quarterly&datatype=csv&apikey={}".format(apikey)
    req = requests.get(url)
    content = req.content
    f = open('data\gdp.csv', 'wb')
    f.write(content)
    f.close()
    return

def treasury_yield(apikey):
    """
    Gets the daily treasury yield for 10 year bonds
    params: apikey (str)
    returns: None
    """
    url = "https://www.alphavantage.co/query?function=TREASURY_YIELD&interval=daily&datatype=csv&apikey={}".format(apikey)
    req = requests.get(url)
    content = req.content
    f = open('data\\treasury.csv', 'wb')
    f.write(content)
    f.close()
    return

def federal_funds_rate(apikey):
    """
    Gets the daily federal funds rate
    params: apikey (str)
    returns: None
    """
    url = "https://www.alphavantage.co/query?function=FEDERAL_FUNDS_RATE&interval=daily&datatype=csv&apikey={}".format(apikey)
    req = requests.get(url)
    content = req.content
    f = open('data\\federal.csv', 'wb')
    f.write(content)
    f.close()
    return

def cpi(apikey):
    """
    Gets the monthly consumer price index
    params: apikey (str)
    returns: None
    """
    url = "https://www.alphavantage.co/query?function=CPI&interval=monthly&datatype=csv&apikey={}".format(apikey)
    req = requests.get(url)
    content = req.content
    f = open('data\cpi.csv', 'wb')
    f.write(content)
    f.close()
    return

def inflation(apikey):
    """
    Gets the annual inflation rate
    params: apikey (str)
    returns: None
    """
    url = "https://www.alphavantage.co/query?function=INFLATION&datatype=csv&apikey={}".format(apikey)
    req = requests.get(url)
    content = req.content
    f = open('data\inflation.csv', 'wb')
    f.write(content)
    f.close()
    return

def unemployment(apikey):
    """
    Gets the monthly unemployment rate 
    params: apikey (str)
    returns: None
    """
    url = "https://www.alphavantage.co/query?function=UNEMPLOYMENT&datatype=csv&apikey={}".format(apikey)
    req = requests.get(url)
    content = req.content
    f = open('data\\unemployment.csv', 'wb')
    f.write(content)
    f.close()
    return


# Technical Indicators
def sma(apikey, symbol):
    """
    Gets the simple moving average (SMA) values for given stock
    params: apikey (str), symbol (str)
    returns: None
    """
    url = "https://www.alphavantage.co/query?function=SMA&symbol={}&interval=daily&time_period=10&series_type=open&datatype=csv&apikey={}".format(symbol, apikey)
    req = requests.get(url)
    content = req.content
    f = open('data\sma_{}.csv'.format(symbol), 'wb')
    f.write(content)
    f.close()
    return

def ema(apikey, symbol):
    """
    Gets the exponential moving average (SMA) values for given stock
    params: apikey (str), symbol (str)
    returns: None
    """
    url = "https://www.alphavantage.co/query?function=EMA&symbol={}&interval=daily&time_period=10&series_type=open&datatype=csv&apikey={}".format(symbol, apikey)
    req = requests.get(url)
    content = req.content
    f = open('data\ema_{}.csv'.format(symbol), 'wb')
    f.write(content)
    f.close()
    return

def stoch(apikey, symbol):
    """
    Gets the stochastic oscillator (STOCH) values for given stock
    params: apikey (str), symbol (str)
    returns: None
    """
    url = "https://www.alphavantage.co/query?function=STOCH&symbol={}&interval=daily&time_period=10&series_type=open&datatype=csv&apikey={}".format(symbol, apikey)
    req = requests.get(url)
    content = req.content
    f = open('data\stoch_{}.csv'.format(symbol), 'wb')
    f.write(content)
    f.close()
    return

def rsi(apikey, symbol):
    """
    Gets the relative strength index (RSI) values for given stock
    params: apikey (str), symbol (str)
    returns: None
    """
    url = "https://www.alphavantage.co/query?function=RSI&symbol={}&interval=daily&time_period=10&series_type=open&datatype=csv&apikey={}".format(symbol, apikey)
    req = requests.get(url)
    content = req.content
    f = open('data\\rsi_{}.csv'.format(symbol), 'wb')
    f.write(content)
    f.close()
    return

def adx(apikey, symbol):
    """
    Gets the average directional movement index (ADX) values for given stock
    params: apikey (str), symbol (str)
    returns: None
    """
    url = "https://www.alphavantage.co/query?function=ADX&symbol={}&interval=daily&time_period=10&datatype=csv&apikey={}".format(symbol, apikey)
    req = requests.get(url)
    content = req.content
    f = open('data\\adx_{}.csv'.format(symbol), 'wb')
    f.write(content)
    f.close()
    return

def cci(apikey, symbol):
    """
    Gets the commodity channel index (CCI) values for given stock
    params: apikey (str), symbol (str)
    returns: None
    """
    url = "https://www.alphavantage.co/query?function=CCI&symbol={}&interval=daily&time_period=10&datatype=csv&apikey={}".format(symbol, apikey)
    req = requests.get(url)
    content = req.content
    f = open('data\cci_{}.csv'.format(symbol), 'wb')
    f.write(content)
    f.close()
    return

def ad(apikey, symbol):
    """
    Gets the Chaikin A/D (AD) line values for given stock
    params: apikey (str), symbol (str)
    returns: None
    """
    url = "https://www.alphavantage.co/query?function=AD&symbol={}&interval=daily&time_period=10&datatype=csv&apikey={}".format(symbol, apikey)
    req = requests.get(url)
    content = req.content
    f = open('data\\ad_{}.csv'.format(symbol), 'wb')
    f.write(content)
    f.close()
    return

def obv(apikey, symbol):
    """
    Gets the on balance value (OBV) line values for given stock
    params: apikey (str), symbol (str)
    returns: None
    """
    url = "https://www.alphavantage.co/query?function=AD&symbol={}&interval=daily&time_period=10&datatype=csv&apikey={}".format(symbol, apikey)
    req = requests.get(url)
    content = req.content
    f = open('data\obv_{}.csv'.format(symbol), 'wb')
    f.write(content)
    f.close()
    return


def get_data(apikey):
    """
    Calls the get_ functions to retrieve the necessary data. Since we are using the free api which is limited
    to 5 calls/minute we need to implement a timer to split the api calls so that we don't go over the api call 
    limit. Then we merge the data into a single dataframe using outer union logic which we write to the current 
    directory as csv file
    params: apikey (str)
    returns: None
    """
    symbols = ["AAPL", "MSFT", "TSLA", "AMZN", "GOOG"]
    time_series = [time_series_daily, time_series_weekly, time_series_monthly]
    fundamental = []
    economic = [real_gdp, treasury_yield, federal_funds_rate, cpi, inflation, unemployment]
    technical = [sma, ema, stoch, rsi, adx, cci, ad, obv]

    for symbol in symbols:
        count = 1
        for f in time_series:
            if count == 5:
                time.sleep(60)
                count = 1
            else:
                f(apikey, symbol)
                count += 1
        
        for f in economic:
            if count == 5:
                time.sleep(60)
                count = 1
            else:
                f(apikey)
                count += 1
        
        for f in technical:
            if count == 5:
                time.sleep(60)
                count = 1
            else:
                f(apikey, symbol)
                count += 1
    return


def date2int(string):
    """
    Utility function that converts a string "YYYY-MM-DD" into an integer YYYYMMDD
    params: string (str)
    returns: output (int)
    """
    output = [i for i in string if i.isdigit()]
    output = "".join(output)
    output = int(output)
    return output


def apple_df():
    """
    Cleans up and formats a dataframe corresponding to relevant data for Apple
    params: None
    return: DataFrame
    """
    daily = pd.read_csv("data/aapl/daily_AAPL.csv")
    daily.rename({'timestamp':'time'}, axis=1, inplace=True)
    ad = pd.read_csv("data/aapl/ad_AAPL.csv")
    adx = pd.read_csv("data/aapl/adx_AAPL.csv")
    ema = pd.read_csv("data/aapl/ema_AAPL.csv")
    obv = pd.read_csv("data/aapl/obv_AAPL.csv")
    rsi = pd.read_csv("data/aapl/rsi_AAPL.csv")
    stoch = pd.read_csv("data/aapl/stoch_AAPL.csv")

    dframes = [daily, ad, adx, ema, obv, rsi, stoch]
    df = functools.reduce(lambda  left,right: pd.merge(left,right,on=['time'], how='outer'), dframes)

    df.dropna(inplace=True)
    df["time"] = df["time"].apply(date2int)
    return df


def googl_df():
    """
    Cleans up and formats a dataframe corresponding to relevant data for Google (Alphabet)
    params: None
    return: DataFrame
    """
    daily = pd.read_csv("data/google/daily_GOOG.csv")
    daily.rename({'timestamp':'time'}, axis=1, inplace=True)
    ad = pd.read_csv("data/google/ad_GOOG.csv")
    adx = pd.read_csv("data/google/adx_GOOG.csv")
    ema = pd.read_csv("data/google/ema_GOOG.csv")
    obv = pd.read_csv("data/google/obv_GOOG.csv")
    rsi = pd.read_csv("data/google/rsi_GOOG.csv")
    stoch = pd.read_csv("data/google/stoch_GOOG.csv")

    dframes = [daily, ad, adx, ema, obv, rsi, stoch]
    df = functools.reduce(lambda  left,right: pd.merge(left,right,on=['time'], how='outer'), dframes)

    df.dropna(inplace=True)
    df["time"] = df["time"].apply(date2int)
    return df


def amzn_df():
    """
    Cleans up and formats a dataframe corresponding to relevant data for Amazon
    params: None
    return: DataFrame
    """
    daily = pd.read_csv("data/amzn/daily_AMZN.csv")
    daily.rename({'timestamp':'time'}, axis=1, inplace=True)
    ad = pd.read_csv("data/amzn/ad_AMZN.csv")
    adx = pd.read_csv("data/amzn/adx_AMZN.csv")
    ema = pd.read_csv("data/amzn/ema_AMZN.csv")
    obv = pd.read_csv("data/amzn/obv_AMZN.csv")
    rsi = pd.read_csv("data/amzn/rsi_AMZN.csv")
    stoch = pd.read_csv("data/amzn/stoch_AMZN.csv")

    dframes = [daily, ad, adx, ema, obv, rsi, stoch]
    df = functools.reduce(lambda  left,right: pd.merge(left,right,on=['time'], how='outer'), dframes)

    df.dropna(inplace=True)
    df["time"] = df["time"].apply(date2int)
    return df


def msft_df():
    """
    Cleans up and formats a dataframe corresponding to relevant data for Microsoft
    params: None
    return: DataFrame
    """
    daily = pd.read_csv("data/msft/daily_MSFT.csv")
    daily.rename({'timestamp':'time'}, axis=1, inplace=True)
    ad = pd.read_csv("data/msft/ad_MSFT.csv")
    adx = pd.read_csv("data/msft/adx_MSFT.csv")
    ema = pd.read_csv("data/msft/ema_MSFT.csv")
    obv = pd.read_csv("data/msft/obv_MSFT.csv")
    rsi = pd.read_csv("data/msft/rsi_MSFT.csv")
    stoch = pd.read_csv("data/msft/stoch_MSFT.csv")

    dframes = [daily, ad, adx, ema, obv, rsi, stoch]
    df = functools.reduce(lambda  left,right: pd.merge(left,right,on=['time'], how='outer'), dframes)

    df.dropna(inplace=True)
    df["time"] = df["time"].apply(date2int)
    return df


def tsla_df():
    """
    Cleans up and formats a dataframe corresponding to relevant data for Tesla
    params: None
    return: DataFrame
    """
    daily = pd.read_csv("data/tsla/daily_TSLA.csv")
    daily.rename({'timestamp':'time'}, axis=1, inplace=True)
    ad = pd.read_csv("data/tsla/ad_TSLA.csv")
    adx = pd.read_csv("data/tsla/adx_TSLA.csv")
    ema = pd.read_csv("data/tsla/ema_TSLA.csv")
    obv = pd.read_csv("data/tsla/obv_TSLA.csv")
    rsi = pd.read_csv("data/tsla/rsi_TSLA.csv")
    stoch = pd.read_csv("data/tsla/stoch_TSLA.csv")

    dframes = [daily, ad, adx, ema, obv, rsi, stoch]
    df = functools.reduce(lambda  left,right: pd.merge(left,right,on=['time'], how='outer'), dframes)

    df.dropna(inplace=True)
    df["time"] = df["time"].apply(date2int)
    return df


def merge_data():
    """
    Merges the data into appropriate multidimensional array for neural network training and saves
    them using pickle
    params: None
    returns: None
    """
    apple = apple_df().to_numpy()
    amzn = amzn_df().to_numpy()
    msft = msft_df().to_numpy()
    tensor = np.stack((apple, amzn, msft))
    np.save("data/tensor.npy", tensor)
    return

if __name__ == "__main__":
    apikey = "S8YIUGVLMYAG3S4E"
    get_data(apikey)
    merge_data()