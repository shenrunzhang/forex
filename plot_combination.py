import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import pandas as pd
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from Threshold import get_threshold

MODEL_T = 'models/lstm_technical_eurusd.h5'
MODEL_F = "models/fundamental_model_eurusd_300epochs.h5"

# data is concatenated, need to separate into technical and fundamental datasets
DATA = 'data/technical+fundamental_data_eurusd.csv'

dataframe = read_csv(DATA, engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')

# cut off first 20 values
dataset = dataset[20:]
dataset = dataset[:, 1:]

# normalize the dataset 
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into two
dataset_t = dataset[:,:9]
dataset_f = dataset[:,[0,9,10,11,12,13,14,15]]

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

def train_test_split(dataset):
    n_cols = dataset.shape[1]

    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    # convert an array of values into a dataset matrix

    look_back = 5
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], look_back, n_cols))
    testX = np.reshape(testX, (testX.shape[0], look_back, n_cols))

    return trainX, testX, testY

_, t_testX, testY = train_test_split(dataset_t)
_, f_testX, _ = train_test_split(dataset_f)

model_t = tf.keras.models.load_model(MODEL_T)
model_f = tf.keras.models.load_model(MODEL_F)
print(t_testX[0:5])
print(t_testX.shape)
testPredict_t = model_t.predict(t_testX)
testPredict_t = np.squeeze(testPredict_t)

testPredict_f = model_f.predict(f_testX)
testPredict_f = np.squeeze(testPredict_f)

# Inverse scale the data

def inverse_transform(arr):
    num_cols_full_dataset = 16
    extended = np.zeros((len(arr), num_cols_full_dataset))
    extended[:, 0] = arr
    return scaler.inverse_transform(extended)[:, 0]

testPredict_t = inverse_transform(testPredict_t)
testPredict_f = inverse_transform(testPredict_f)
print("testPredict shape", testPredict_f.shape)

testY = inverse_transform(testY)

# threshold and make decisions
threshold = get_threshold(dataframe["Close"]) * 3


def decision(diff):
    if diff > threshold:
        return "I"
    if -diff > threshold:
        return "D"
    else:
        return "N"

def union(a, b):
    if a == "I" and b == "I":
        return "C"
    if a == "D" and b == "D":
        return "C"
    if a == "N" or b == "N":
        return "N"
    else:
        return "F"
    
def combination(t_decision, f_decision, t_diff, f_diff, t_loss, f_loss):
    if t_decision == "N" or f_decision == "N":
        return "N"
    if t_decision == f_decision:
        return t_decision
    else:
        # # if conflict, return lowest change
        # if abs(t_diff) > abs(f_diff):
        #     return f_decision
        # else: 
        #     return t_decision
        
        # if conflict, return lowest loss in previous step
        if np.isnan(t_loss) or np.isnan(f_loss):
            return t_decision
        if abs(t_loss) == abs(f_loss):
            return t_decision
        else:
            if abs(t_loss) < abs(f_loss):
                return t_decision
            else:
                return f_decision

table = pd.DataFrame([testY, testPredict_t, testPredict_f]).transpose()
table.columns = ["actual", "technical", "fundamental"]
table["t decision"] = (table.technical - table.actual).apply(decision)
table["f decision"] = (table.fundamental - table.actual).apply(decision)
table["correct"] = table["actual"].diff(
).shift(-1).apply(lambda x: "I" if x > 0 else "D" if x < 0 else "N")
table["t union"] = table.apply(lambda x: union(
    x["t decision"], x["correct"]), axis=1)
table["f union"] = table.apply(lambda x: union(
    x["f decision"], x["correct"]), axis=1)
table["t loss"] = table.technical.shift(1) - table.actual
table["f loss"] = table.fundamental.shift(1) - table.actual
table["t diff"] = table.technical - table.actual
table["f diff"] = table.fundamental - table.actual
table["combination"]  = table.apply(lambda x: combination(
    x["t decision"], x["f decision"], x["t diff"], x["f diff"], x["t loss"], x["f loss"]), axis=1)

table["union"] = table.apply(lambda x: union(
    x["combination"], x["correct"]), axis=1)

print(table)
counts = table["union"].value_counts()
frequency = table["union"].value_counts(normalize=True)
print(pd.DataFrame({"counts": counts, "frequency": frequency}))

# shift predictions up by one
testPredict_t = np.delete(testPredict_t, -1)
testPredict_f = np.delete(testPredict_f, -1)
testY = np.delete(testY, 0)

# Plot results
plt.plot(testPredict_t, color="blue")
plt.plot(testPredict_f, color="green")
plt.plot(testY, color="red")
plt.show()
