
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from six.moves import cPickle as pickle

#Read Input parameters (command line flags)

parser = argparse.ArgumentParser(description='Format input data from metatrader for neural network training')
parser.add_argument('--csv', type=str, required=True, help='Input csv file from metatrader')
parser.add_argument('--lookback', type=int, default=20, help='Generate dataset with lookback into the time series, i.e, how many days for a sample')
parser.add_argument('--randomize', type=int, default=1, help='Randomize dataset')
parser.add_argument('--out', type=str, default='out.pickle', help='Output pickle file')
parser.add_argument('--ratio', type=float, default=0.8, help='Number of samples in the training dataset')

args = parser.parse_args()

outfilename = args.out

#Read data
dataset = pd.read_csv(args.csv, skipfooter=1,
                     engine='python', usecols=[2,3,4,5,6])
dataset = np.array(dataset)
print(dataset.shape)

# print(dataset)
# plt.plot(dataset)
# plt.show()

# In[ ]:


# from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler(feature_range=(0,1))
# dataset = scaler.fit_transform(dataset)

# plt.plot(dataset[1:]/dataset[:-1])
# plt.show()


# In[ ]:

def createDataset(timeseries, lookback):
    X, Y = [], []
    for i in range(len(timeseries) - lookback - 1):
        X.append(timeseries[i:(i + lookback)])
        Y.append(timeseries[i + lookback])
    X = np.array(X)
    return np.array(X), np.array(Y)

lookback = args.lookback
X, Y = createDataset(dataset, lookback)

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

if args.randomize:
    print("Randomizing...")
    X, Y = unison_shuffled_copies(X, Y)

print(X.shape)
print(Y.shape)

lenindex = int(len(X)*args.ratio)

data = {}
data['train_dataset'] = X[0:lenindex]
data['train_labels'] = Y[0:lenindex]
data['test_dataset'] = X[lenindex:]
data['test_labels'] = Y[lenindex:]

# In[ ]:
print('train_dataset', data['train_dataset'].shape)
print('train_labels', data['train_labels'].shape)
print('test_dataset', data['test_dataset'].shape)
print('test_labels', data['test_labels'].shape)

try:
	with open(outfilename, 'wb') as f:
		pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
except Exception as e:
	print('Unable to save data to', set_filename, ':', e)

#Save data



# In[ ]:


# size = int(Y.shape[0] * 2 / 3)
# trainX, trainY = X[0:size], Y[0:size]
# testX, testY = X[size:], Y[size:]

# print(trainX.shape, trainY.shape)
# print(testX.shape, testY.shape)


# In[ ]:


# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM

# model = Sequential()
# model.add(LSTM(10, input_shape=(lookback, 1), return_sequences=True))
# model.add(LSTM(10))
# model.add(Dense(1))

# model.summary()

# model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
# model.fit(trainX, trainY, epochs=1000, batch_size=100, verbose=2)


# # In[ ]:


# model.reset_states()
# trainP = model.predict(trainX)
# testP = model.predict(testX)

# plt.plot(scaler.inverse_transform(dataset))
# plt.plot(np.append(np.repeat(None, lookback), scaler.inverse_transform(trainP)))
# plt.plot(np.append(np.repeat(None, lookback + trainP.shape[0]), scaler.inverse_transform(testP)))
# plt.show()

