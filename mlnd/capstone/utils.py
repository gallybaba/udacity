import numpy as np
import tensorflow as tf
import pandas as pd
import re
import numpy.core.defchararray as npc
from IPython.display import display
import matplotlib
from matplotlib import pyplot as plt
from tensorflow import summary
from sklearn import preprocessing

DATA_FILE="data.csv"
HEADER_FILE="header.csv"
REQIDS_FILE="reqids.csv"

def load_data():
    #print("loading header")
    header = pd.read_csv(HEADER_FILE)
    header.columns = [s.strip() for s in header.columns.values]
    h = header.columns.values
    #print("loading dataset")
    raw_data = pd.read_csv(DATA_FILE, names=h, dtype = {'Date': str, 'ReqId': np.int, 'Open' : np.float64\
        , 'Close' : np.float64, 'High' : np.float64, 'Low' : np.float64, 'Volume' : np.float64})
    #print("loading reqids")
    reqids = pd.read_csv(REQIDS_FILE)
    return header, raw_data, reqids

#print(pd.__name__, pd.__version__)
import pandas.core
from pandas.core import window
from pandas.core.window import EWM

def preprocess_raw_data(raw_data, reqids):
    dcol = raw_data['Date']
    dcolf = dcol.str.match("finished-")
    raw_data_wo_finish = raw_data[~dcolf]
    rownum = raw_data_wo_finish.shape[0]
    raw_data_wo_finish['Date'] = pd.DatetimeIndex(pd.to_datetime(raw_data_wo_finish['Date'], format="%Y%m%d %H:%M:%S"))
    raw_data_wo_finish['ReqId'] = raw_data_wo_finish['ReqId'].astype(int)
    raw_data_wo_finish['Open'] = raw_data_wo_finish['Open'].astype(float)
    raw_data_wo_finish['High'] = raw_data_wo_finish['High'].astype(float)
    raw_data_wo_finish['Low'] = raw_data_wo_finish['Low'].astype(float)
    raw_data_wo_finish['Close'] = raw_data_wo_finish['Close'].astype(float)
    raw_data_wo_finish = pd.merge(raw_data_wo_finish, reqids, on=['ReqId'])
    raw_data_wo_finish = pd.get_dummies(data = raw_data_wo_finish, columns = ['QuoteType', 'Symbol'])
    del raw_data_wo_finish['Volume']
    del raw_data_wo_finish['HasGaps']
    del raw_data_wo_finish['WAP']
    del raw_data_wo_finish['Count']
    del raw_data_wo_finish['ReqId']
    ### lets add more features
    ### 5 period moving average MA
    ### 21 period moving average with bollinger bands bb
    ### 21 period min
    ### 21 period max
    ### TODO:
    ### these periods could be parameterized to describe and evaluate models later on
    raw_data_wo_finish['ShortEMA'] = np.round(raw_data_wo_finish['Close'].ewm(span=5).mean(), 3)
    raw_data_wo_finish['LongEMA']  = np.round(raw_data_wo_finish['Close'].ewm(span=21).mean(), 3)
    raw_data_wo_finish['BBAUpper'] = np.round(2 * raw_data_wo_finish['Close'].ewm(span=21).std() + raw_data_wo_finish['Close'], 3)
    raw_data_wo_finish['BBALower'] = np.round(raw_data_wo_finish['Close'] - 2 * raw_data_wo_finish['Close'].ewm(span=21).std(), 3)
    raw_data_wo_finish['LastMin']  = np.round(raw_data_wo_finish['Close'].rolling(window=21).min(), 3)
    raw_data_wo_finish['LastMax']  = np.round(raw_data_wo_finish['Close'].rolling(window=21).max(), 3)
    ### remove first 20 rows
    raw_data_wo_finish = raw_data_wo_finish.loc[20:,:]
    raw_data_wo_finish = raw_data_wo_finish.reset_index()
    #print(raw_data_wo_finish.head())
    return raw_data_wo_finish

def plot_raw(raw_data):
    plt.figure(1)
    fig, axs = plt.subplots(4, sharex=True)
    raw_data.plot()

    xcoord = np.arange(7)
    xcoord_labels = ['mean', 'std', 'min', '25%', '50%', '75%', 'max']
    raw_describe_open = raw_data.describe()['Open'][1:,]
    axs[0].plot(xcoord, raw_describe_open[:])
    axs[0].set_title("Open Descriptive Stats")
    axs[0].set_xticklabels(xcoord_labels)

    raw_describe_high = raw_data.describe()['High'][1:,]
    axs[1].plot(xcoord, raw_describe_high[:])
    axs[1].set_title("High Descriptive Stats")

    raw_describe_close = raw_data.describe()['Close'][1:,]
    axs[2].plot(xcoord, raw_describe_close[:])
    axs[2].set_title("Close Descriptive Stats")
    xcoord = np.arange(7)
    xcoord_labels = ['mean', 'std', 'min', '25%', '50%', '75%', 'max']
    raw_describe_low = raw_data.describe()['Low'][1:,]
    axs[3].plot(xcoord, raw_describe_low[:])
    axs[3].set_title("Low Descriptive Stats")

def plot_data(data):
    plt.figure(2)
    fig, axs = plt.subplots(4, sharex=True)
    xcoord = np.arange(7)
    xcoord_labels = ['mean', 'std', 'min', '25%', '50%', '75%', 'max']
    data_describe_open = data.describe()['Open'][1:,]
    axs[0].plot(xcoord, data_describe_open[:])
    axs[0].set_title("Open Descriptive Stats")
    axs[0].set_xticklabels(xcoord_labels)

    data_describe_high = data.describe()['High'][1:,]
    axs[1].plot(xcoord, data_describe_high[:])
    axs[1].set_title("High Descriptive Stats")

    data_describe_close = data.describe()['Close'][1:,]
    axs[2].plot(xcoord, data_describe_close[:])
    axs[2].set_title("Close Descriptive Stats")

    data_describe_low = data.describe()['Low'][1:,]
    axs[3].plot(xcoord, data_describe_low[:])
    axs[3].set_title("Low Descriptive Stats")

    bbdata = data.loc[:,['Close', 'BBAUpper', 'BBALower', 'ShortEMA', 'LongEMA'] ]
    bbdata = bbdata[-50:]
    bbdata.plot()

    bbdata2 = data.loc[:,['Close', 'BBAUpper', 'BBALower', 'ShortEMA', 'LongEMA'] ]
    bbdata2 = bbdata2.loc[0:50,:]
    bbdata2.plot()


def generate_labels(training_data, lookahead, range_cl, range_aapl):
    end = training_data.shape[0]
    #print('end: ', end)
    labels = np.zeros((end, 1))
    for begin in np.arange(end):
        #if begin % 5000 == 0:
        #    print('progress...', begin, ' records')
        #print("begin:", begin)
        close = training_data.iloc[begin, 3]
        #print('close at begin:', close)
        look_ahead_begin = begin + 1
        look_ahead_end = (look_ahead_begin + lookahead) % end
        #print('look_ahead_end: ', look_ahead_end)
        look_ahead_data = training_data.loc[look_ahead_begin:look_ahead_end, ['Open', 'Close', 'High', 'Low', 'Symbol_AAPL', 'Symbol_CLZ16']]
        for rownum in range(look_ahead_begin,look_ahead_end):
            #print('rownum: ', rownum)
            row = look_ahead_data.loc[rownum,:]
            #print('got row: ', row)
            max_price = np.max(row)
            #print('maxprice: ', max_price)
            if row['Symbol_CLZ16'] == 1:
                if np.abs(close - max_price) >= range_cl:
                    #print('range in CL: ', np.abs(close - max_price), ' Close: ', close, ' Max: ', max_price)
                    labels[begin] = 1
                #else:
                #    labels[begin][1] = 1
            elif row['Symbol_AAPL'] == 1:
                if np.abs(close - max_price) >= range_aapl:
                    #print('range in AAPL: ', np.abs(close - max_price), ' Close: ', close, ' Max: ', max_price)
                    labels[begin] = 1
                #else:
                #    labels[begin][1] = 1

    #print('found ', np.sum(labels), 'possible trades')
    with tf.name_scope("labels"):
        summary.scalar("LookAheadPeriod", lookahead)
        summary.scalar("RangeCL", range_cl)
        summary.scalar("RangeAAPL", range_aapl)
        summary.scalar("PotentialTrades", np.sum(labels))
    return labels

def scale(training_data):
    return preprocessing.scale(training_data)

def split_data(standardized_data, training_labels):
    total_rows, total_cols = standardized_data.shape[0], standardized_data.shape[1]
    train_len = int(0.6 * total_rows)
    validate_len = int(0.2 * total_rows)
    test_len = int(0.2 * total_rows)

    std_train_data = standardized_data[0:train_len]
    std_train_label = training_labels[0:train_len, 0]
    #print("Train positives: ", np.sum(std_train_label))
    std_validate_data = standardized_data[train_len+1:train_len + 1 + validate_len]
    std_validate_label = training_labels[train_len+1:train_len + 1 + validate_len, 0]
    #print("Validate positives: ", np.sum(std_validate_label))
    test_start = train_len + validate_len + 1
    std_test_data = standardized_data[test_start:test_start + test_len]
    std_test_label = training_labels[validate_len+1:validate_len + 1 + test_len, 0]
    #print("Test positives: ", np.sum(std_test_label))
    #print('training len: ', train_len, ', validate len: ', validate_len, ', \
    #test len: ', test_len, ', total rows: ', total_rows)
    #print('training shape: ', std_train_data.shape, std_train_label.shape)
    #print('validation shape: ', std_validate_data.shape, std_validate_label.shape)
    #print('test shape: ', std_test_data.shape, std_test_label.shape)
    return std_train_data, std_train_label, std_validate_data, std_validate_label, std_test_data, std_test_label


