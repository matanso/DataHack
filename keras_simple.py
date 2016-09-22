from __future__ import print_function
import numpy as np

np.random.seed(1337)  # for reproducibility

import pandas as pd
import matplotlib.pyplot as plt
import os
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import pickle


batch_size = 256
nb_epoch = 3

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)
#
# the_dir = 'data'
# train_df = pd.read_csv(os.path.join(the_dir, 'taxi.train.csv.gz'), compression='gzip')
# valid_df = pd.read_csv(os.path.join(the_dir, 'taxi.valid.csv.gz'), compression='gzip')
# test_df = pd.read_csv(os.path.join(the_dir, 'taxi.test.no.label.csv.gz'), compression='gzip')
#
#
#
#
# def conv_date(d):
#     d = d.split()[1].split(':')
#     return (int(d[0]) * 60 + int(d[1])) * 60 + int(d[2])
#
#
# def convert_dates(arr):
#     return np.array([conv_date(i) for i in arr])
#
#
# def get_x_y(df):
#     return np.array([np.array(df['to_latitude']),
#                      np.array(df['to_longitude']),
#                      np.array(df['from_latitude']),
#                      np.array(df['from_longitude']),
#                      np.array(df['trip_distance']),
#                      convert_dates(df['from_datetime'])]).transpose(), np.array(df['y'])
#
#
# # the data, shuffled and split between train and test sets
# (X_train, Y_train), (X_test, Y_test) = get_x_y(train_df), get_x_y(valid_df)
#
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
#
#
# def ride_filter(ride, y):
#     y = 10 ** y
#     return 3600 > y / ride[4] > 25
#
#
# train_filter_arr = np.array([ride_filter(X_train[i], Y_train[i]) for i in range(X_train.shape[0])])
# X_train = np.array([X_train[i] for i in range(X_train.shape[0]) if train_filter_arr[i]])
# Y_train = np.array([Y_train[i] for i in range(Y_train.shape[0]) if train_filter_arr[i]])
#
# pc = [np.percentile(X_train[:, i], [9, 91]) for i in range(X_train.shape[1])]
#
# for t in X_train, X_test:
#     for i in range(t.shape[1]):
#         t[:, i] -= pc[i][0]
#         t[:, i] /= pc[i][1] - pc[i][0]
#
# print('X_train shape:', X_train.shape)
# print(X_train.shape[0], 'train samples')
# print(X_test.shape[0], 'test samples')
# print(X_train[0])
(X_train, Y_train, X_test, Y_test) = pickle.load(open('data/stuff.pck', 'r'))
# model = Sequential()
#
# #X_train, X_test = X_train[:, -2:], X_test[:, -2:]
#
# model.add(Dense(128, input_shape=X_train.shape[1:]))
# model.add(Activation('relu'))
# model.add(Dense(256))
# model.add(Activation('relu'))
# model.add(Dense(256))
# model.add(Activation('relu'))
# model.add(Dense(256))
# model.add(Activation('relu'))
# model.add(Dense(256))
# model.add(Activation('relu'))
# model.add(Dense(256))
# model.add(Activation('relu'))
# model.add(Dense(256))
# model.add(Activation('relu'))
# model.add(Dense(256))
# model.add(Activation('relu'))
# model.add(Dense(256))
# model.add(Activation('relu'))
# model.add(Dense(256))
# model.add(Activation('relu'))
# model.add(Dense(256))
# model.add(Activation('relu'))
# model.add(Dropout(0.25))
# model.add(Dense(1))
#
# model.compile(loss='mean_squared_error',
#               optimizer='adadelta',
#               metrics=['accuracy'])
#
# model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
#           verbose=1, validation_data=(X_test, Y_test))


from keras.models import load_model

model = load_model('models/all_params_10_layers.hd5')

x = np.array([np.array([i / 1000, 0.3, i / 1000 + 0.001, 0.4, 0.3, 0.5]) for i in range(1000)])
y = model.predict(x)


plt.plot(x[:, 5], y, 'ro')
plt.show()
raw_input()
# model.save('models/all_params_10_layers.hd5')
score = model.evaluate(X_test, Y_test, verbose=0)

print('Test score:', score[0])
print('Test accuracy:', score[1])
