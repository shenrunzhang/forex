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

MODEL = 'prob_model.h5'
DATA = r'C:\Users\Shen\Documents\forex\technical_data_eurusd2.csv'

dataframe = read_csv(DATA, engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')

# cut off first 20 values
dataset = dataset[20:]
dataset = dataset[:, 1:]

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

# convert an array of values into a dataset matrix


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


look_back = 5
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], look_back, 9))
testX = np.reshape(testX, (testX.shape[0], look_back, 9))

model = tf.keras.models.load_model(MODEL)

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

trainPredict = np.squeeze(trainPredict)
testPredict = np.squeeze(testPredict)

# Inverse scale the data


def inverse_transform(arr):
    extended = np.zeros((len(arr), 9))
    extended[:, 0] = arr
    return scaler.inverse_transform(extended)[:, 0]


trainPredict = inverse_transform(trainPredict)
testPredict = inverse_transform(testPredict)
trainY = inverse_transform(trainY)
testY = inverse_transform(testY)

prob = model.predict_proba(testX)
print("testY", testY)
print("test predict", testPredict)
print("probab", prob)


# # threshold and make decisions
# threshold = get_threshold(dataframe["Close"])


# def decision(diff):
#     if diff > threshold:
#         return "I"
#     if -diff > threshold:
#         return "D"
#     else:
#         return "N"


# def union(a, b):
#     if a == "I" and b == "I":
#         return "C"
#     if a == "D" and b == "D":
#         return "C"
#     if a == "N" or b == "N":
#         return "N"
#     else:
#         return "F"


# table = pd.DataFrame([testY, testPredict]).transpose()
# table.columns = ["actual", "predict"]
# table["p-a"] = table.predict - table.actual
# table["decision"] = table["p-a"].apply(decision)
# table["correct"] = table["actual"].diff(
# ).shift(-1).apply(lambda x: "I" if x > 0 else "D" if x < 0 else "N")
# table["union"] = table.apply(lambda x: union(
#     x["decision"], x["correct"]), axis=1)
# counts = table["union"].value_counts()
# frequency = table["union"].value_counts(normalize=True)
# print(pd.DataFrame({"counts": counts, "frequency": frequency}))

# shift predictions up by one
testPredict = np.delete(testPredict, -1)
testY = np.delete(testY, 0)

# Plot results
plt.plot(testPredict, color="blue")
plt.plot(testY, color="red")
plt.plot(prob, color="green")
plt.show()
testScore = np.sqrt(mean_squared_error(testY, testPredict))
print('Test Score: %.2f RMSE' % (testScore))

