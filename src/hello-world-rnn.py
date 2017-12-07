from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import BaseLogger
from math import sqrt
from matplotlib import pyplot
import numpy as np

# date-time parsing function for loading the dataset
def parser(x):
    try:
        return datetime.strptime('190'+x, '%Y-%m')
    except:
        return datetime.strptime(x, '%Y.%m.%d')

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df

# create a differenced series
def difference(dataset, interval=1):
    # diff = list()
    # for i in range(interval, len(dataset)):
    #     value = dataset[i] - dataset[i - interval]
    #     diff.append(value)
    # return Series(diff)
    return np.diff(dataset, axis=0)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

# scale train and test data to [-1, 1]
def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled

# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]

# fit an LSTM network to training data
def fit_lstm(X, y, batch_size, nb_epoch, neurons):
    #X, y = train[:, 0:-1], train[:, -1]
    # X = X.reshape(X.shape[0], 1, X.shape[1])
    # print('X=', X, 'y=', y)
    print('X=', X.shape, 'y=', y.shape)
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(X.shape[2]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        print("epoch:", i)
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False, callbacks=[BaseLogger()])
        model.reset_states()
    return model

# make a one-step forecast
# def forecast_lstm(model, batch_size, X):
#     X = X.reshape(1, 1, len(X))
#     yhat = model.predict(X, batch_size=batch_size)
#     return yhat[0,0]

# raw_dataset = read_csv("WTICrude1440.csv", skipfooter=1,
#                      engine='python', usecols=[2,3,4,5,6])
raw_dataset = read_csv("WTICrude1440.csv", skipfooter=1,
                     engine='python', usecols=[2,3,4,5,6])
raw_dataset = np.array(raw_dataset)

dataset = raw_dataset
#dataset = np.diff(raw_dataset, axis=0)
dataset = np.gradient(raw_dataset, axis=0)
scaler = MinMaxScaler(feature_range=(-1, 1))
dataset = scaler.fit_transform(dataset)


lookback = 1
X, Y = [], []
for i in range(len(dataset) - lookback):
  X.append(dataset[i:(i + lookback)])
  Y.append(dataset[i + lookback])
X = np.array(X).astype(np.float32)
Y = np.array(Y).astype(np.float32)

size = dataset.shape

# for x, y in zip(X, Y):
#     print("x=", x, "y=", y)

num_train = int(len(X)*.8)
train_dataset = X[0:num_train]
train_labels = Y[0:num_train]
test_dataset = X[num_train:-1]
test_labels = Y[num_train:-1]
test_labels_scaled = raw_dataset[(num_train + lookback):-1]

batch_size = 1


# fit the model
lstm_model = fit_lstm(train_dataset, train_labels, 1, 20, 128)
# forecast the entire training dataset to build up state for forecasting
#train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
lstm_model.predict(train_dataset, batch_size=batch_size)

# walk-forward validation on the test data


# for i, X in enumerate(test_dataset):
    # make one-step forecast
    #X_ = X.reshape(((1,) + X.shape))
Yhat = lstm_model.predict(test_dataset, batch_size=batch_size)
#yhat = yhat[0:batch_size]
# invert scaling
#yhat = scaler.inverse_transform(yhat)
# invert differencing
#yhat = yhat + raw_dataset[num_train + lookback + i]
# store forecast
for i, yhat in enumerate(Yhat):
    print("Predicted:", yhat, "Expected", test_labels[i])

# report performance
rmse = sqrt(mean_squared_error(test_labels, Yhat))
print('Test RMSE: %.3f' % rmse)

predictions_scaled = 2.0*scaler.inverse_transform(Yhat)
predictions_scaled[0] = raw_dataset[num_train + lookback - 1] + predictions_scaled[0]
for i in range(1, len(predictions_scaled)):
  predictions_scaled[i] += predictions_scaled[i - 1]

#predictions_scaled += raw_dataset[num_train + lookback -1:-2]
# predictions_scaled = predictions
# predictions_scaled = scaler.inverse_transform(predictions)

rmse = sqrt(mean_squared_error(test_labels_scaled, predictions_scaled))
print('Test RMSE: %.3f' % rmse)

real_prediction = []
nextlabel = test_labels[-1]
for i in range(50):
    nextlabel = nextlabel.reshape(1,lookback, size[1])
    nextlabel = lstm_model.predict(nextlabel, batch_size=batch_size)
    real_prediction.append(nextlabel.reshape(size[1]))

real_prediction = 2.0*scaler.inverse_transform(real_prediction)
real_prediction[0] = predictions_scaled[-1] + real_prediction[0]
for i in range(1, len(real_prediction)):
  real_prediction[i] += real_prediction[i - 1]
# line plot of observed vs predicted
pyplot.plot(test_labels_scaled[:,0])
pyplot.plot(predictions_scaled[:,0])
pyplot.plot(range(len(predictions_scaled),len(predictions_scaled) + len(real_prediction)), real_prediction[:,0])
pyplot.show()