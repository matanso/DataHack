import numpy as np

np.random.seed(None)  # for reproducibility

import pandas as pd
import numpy as np

import sklearn.ensemble
import sklearn.cluster
import sklearn.metrics
import sklearn.linear_model
import matplotlib.pyplot as plt
import os

k = 100


def conv_date(d):
    d = d.split()[1].split(':')
    return (int(d[0]) * 60 + int(d[1])) * 60 + int(d[2])


def convert_dates(arr):
    return np.array([conv_date(i) for i in arr])


# Loading data
the_dir = '.'
train_df = pd.read_csv(os.path.join(the_dir, 'data/taxi.train.csv.gz'), nrows=2000000, compression='gzip')
valid_df = pd.read_csv(os.path.join(the_dir, 'data/taxi.valid.csv.gz'), nrows=1000000, compression='gzip')

# Cleaning data
train_df = train_df[(train_df['from_longitude'] <= -73.7)]
train_df = train_df[(train_df['from_latitude'] >= 40.5)]
train_df = train_df[(train_df['from_latitude'] <= 41)]
train_df = train_df[((10 ** train_df['y']) / train_df['trip_distance']) <= 2000]
train_df = train_df[((10 ** train_df['y']) / train_df['trip_distance']) >= 40]
train_df = train_df.dropna(axis=0)

train_df['from_datetime'] = convert_dates(train_df['from_datetime'])
valid_df['from_datetime'] = convert_dates(valid_df['from_datetime'])

# get important paramenters
clust_columns = ['from_longitude', 'from_latitude']
reg_columns = ['trip_distance','from_datetime']
X_train = train_df[clust_columns]
X_valid = valid_df[clust_columns]

n = len(X_train)

# Clustering by X values
kmeans = sklearn.cluster.KMeans(n_clusters=k, max_iter=3000)
kmeans.fit(X_train)

labels = kmeans.labels_
train_df['label'] = labels

# plot colored points
plt.scatter(x=X_train['from_longitude'], y=X_train['from_latitude'], c=labels)
# plt.show()


# Running regression for each cluster
regressors = [None] * k

models = [sklearn.ensemble.RandomForestRegressor(n_estimators=100), sklearn.linear_model.LinearRegression(),
          sklearn.ensemble.GradientBoostingRegressor(),
          sklearn.ensemble.ExtraTreesRegressor(n_estimators=10), sklearn.ensemble.BaggingRegressor(),
          sklearn.ensemble.AdaBoostRegressor()]

for label in range(k):
    cur_train_df = train_df[(train_df['label'] == label)]
    # get important paramenters
    X_train, y_train = cur_train_df[reg_columns], cur_train_df.y.values

    #regressors[label] = np.random.choice(models)
    #print regressors[label]
    #regressors[label] = sklearn.ensemble.RandomForestRegressor(n_estimators=100)
    regressors[label] = sklearn.ensemble.AdaBoostRegressor()
    regressors[label].fit(X_train, y_train)

# predicting stuff
groups = kmeans.predict(X_valid)
valid_df['label'] = groups

global_y_valid = []
global_y_pred = []

for label in range(k):
    cur_valid_df = valid_df[(valid_df['label'] == label)]
    # get important paramenters
    X_valid, y_valid = cur_valid_df[reg_columns], cur_valid_df.y.values
    y_pred = regressors[label].predict(X_valid)
    # y_pred = np.log10(y_pred)

    global_y_valid += y_valid.tolist()
    global_y_pred += y_pred.tolist()

print sklearn.metrics.mean_squared_error(global_y_valid, global_y_pred)

print "ya"
