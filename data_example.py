import itertools
import numpy as np
import sklearn.ensemble

import pandas as pd
import matplotlib.pyplot as plt
import os

the_dir = 'data'
train_df = pd.read_csv(os.path.join(the_dir, 'taxi.train.csv.gz'), compression='gzip')
valid_df = pd.read_csv(os.path.join(the_dir, 'taxi.valid.csv.gz'), compression='gzip')
test_df = pd.read_csv(os.path.join(the_dir, 'taxi.test.no.label.csv.gz'), compression='gzip')
# print test_df['trip_distance']
columns = ['passenger_count', 'trip_distance']
fields = ['passenger_count',
          'to_latitude',
          'to_longitude',
          'from_latitude',
          'from_longitude',
          'trip_distance',
          'PaymentFactor',
          'RatecodeFactor',
          'VendorId',
          'from_datetime',
          'y']

plt.plot(train_df['from_latitude'], train_df['from_longitude'], 'ro')
plt.show()
raw_input()
