from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

# Using pandas to read the CSV file.
import pandas as pd
import numpy as np
import tensorflow as tf
import glob
import os
from numpy import nan

# Removed the csv header as part of the ETL proces so we'll define them here.
names = [
    'FL_DATE', 
    'UNIQUE_CARRIER', 
    'AIRLINE_ID', 
    'CARRIER', 
    'FL_NUM',
    'ORIGIN_AIRPORT_ID',
    'ORIGIN_AIRPORT_SEQ_ID',
    'ORIGIN_CITY_MARKET_ID',
    'ORIGIN',
    'DEST_AIRPORT_ID',
    'DEST_AIRPORT_SEQ_ID',
    'DEST_CITY_MARKET_ID',
    'DEST',
    'CRS_DEP_TIME',
    'DEP_TIME',
    'DEP_DELAY',
    'TAXI_OUT',
    'WHEELS_OFF',
    'WHEELS_ON',
    'TAXI_IN',
    'CRS_ARR_TIME',
    'ARR_TIME',
    'ARR_DELAY',
    'CANCELLED',
    'CANCELLATION_CODE',
    'DIVERTED',
    'DISTANCE'
]

# Here we'll specify the dtypes.
dtypes = {
    'FL_DATE': str,
    'UNIQUE_CARRIER': str,
    'AIRLINE_ID': np.float64,
    'CARRIER': str, 
    'FL_NUM': np.float32, 
    'ORIGIN_AIRPORT_ID': np.float32, 
    'ORIGIN_AIRPORT_SEQ_ID': np.float32,
    'ORIGIN_CITY_MARKET_ID': np.float32, 
    'ORIGIN': str, 
    'DEST_AIRPORT_ID': np.float32, 
    'DEST_AIRPORT_SEQ_ID': np.float32, 
    'DEST_CITY_MARKET_ID': np.float32, 
    'DEST': str, 
    'CRS_DEP_TIME': np.float32, 
    'DEP_TIME': np.float32, 
    'DEP_DELAY': np.float32, 
    'TAXI_OUT': np.float32,
    'WHEELS_OFF': np.float32,
    'WHEELS_ON': np.float32,
    'TAXI_IN': np.float32,
    'CRS_ARR_TIME': np.float32,
    'ARR_TIME': np.float32,
    'ARR_DELAY': np.float32,
    'CANCELLED': np.float32,
    'CANCELLATION_CODE': str,
    'DIVERTED': np.float32,
    'DISTANCE': np.float32, 
}

path = '/Users/Nick/Git/data-science-on-gcp/02_ingest/flights/01/201701.csv' # use your path
# allFiles = glob.glob(path + "01/*.csv")
# print("printing allfiles")
# print(allFiles)
# frame = pd.DataFrame()
# list_ = []
# for file_ in allFiles:
#     print(file_)
#     df = pd.read_csv(file_,index_col=None, header=0)
#     list_.append(df)
# frame = pd.concat(list_)

# Read the file.

#df = pd.concat([pd.read_csv(f) for f in glob.glob(path +'*.csv')], names=names, ignore_index = True)
df = pd.read_csv(path, header=0, skipinitialspace=True, names=names)
print('Dataframe dimensions:', df.shape)

tab_info=pd.DataFrame(df.dtypes).T.rename(index={0:'column type'})
tab_info=tab_info.append(pd.DataFrame(df.isnull().sum()).T.rename(index={0:'null values (nb)'}))
tab_info=tab_info.append(pd.DataFrame(df.isnull().sum()/df.shape[0]*100).T.rename(index={0:'null values (%)'}))
tab_info

df.fillna("0", inplace=True)
df['TAXI_OUT'] = df['TAXI_OUT'].apply(pd.to_numeric)
df['DEP_TIME'] = df['DEP_TIME'].apply(pd.to_numeric)
#df['ARR_DELAY'] = df['ARR_DELAY'].apply(pd.to_numeric)

print(df.index)
print(df.columns)

#print(pd.DataFrame(df))

df.info()

# Split the data into a training set and an eval set.

training_data = df

#eval_data = df.iloc[1]
#test_data = df.iloc[:10]

#dataset = tf.data.Dataset.from_tensor_slices((features, labels))

#print(training_data)

#training_data, training_label = df, df.pop(ARR_TIME)
#training_label = pd.DataFrame(df, columns=['ARR_DELAY'])
training_label = df['DEST_AIRPORT_ID']
#eval_label = eval_data.pop('FL_NUM')
#test_label = test_data.pop('FL_NUM')

training_input_fn = tf.estimator.inputs.pandas_input_fn(x=training_data, y=training_label, batch_size=0, shuffle=False)

#eval_input_fn = tf.estimator.inputs.pandas_input_fn(x=eval_data, y=eval_label, batch_size=64, shuffle=False)

#test_input_fn = tf.estimator.inputs.pandas_input_fn(x=test_data, y=test_label, batch_size=10, shuffle=False)

#Feature columns
#carrier = tf.feature_column.categorical_column_with_vocabulary_list('CARRIER', vocabulary_list=['WN', 'OO', 'NK', 'AA', 'DL', 'UA'])
distance = tf.feature_column.numeric_column('DISTANCE')
dep_time = tf.feature_column.numeric_column('DEP_TIME')
taxi_out = tf.feature_column.numeric_column('TAXI_OUT')
fl_num = tf.feature_column.numeric_column('FL_NUM')
#Linear Regressor

linear_features = [distance, dep_time, taxi_out, fl_num]
regressor = tf.estimator.LinearRegressor(feature_columns=linear_features)
regressor.train(input_fn=training_input_fn, steps=10000)
#regressor.evaluate(input_fn=eval_input_fn)

#Deep Neural Network

dnn_features = [
    #numerical features
    distance, dep_time, taxi_out, fl_num,
    # densify categorical features:
    tf.feature_column.indicator_column(carrier),
]

dnnregressor = tf.contrib.learn.DNNRegressor(feature_columns=dnn_features, hidden_units=[50, 30, 10])
dnnregressor.fit(input_fn=training_input_fn, steps=10000)
dnnregressor.evaluate(input_fn=eval_input_fn)

#Predict

predictions = list(dnnregressor.predict_scores(input_fn=training_input_fn))
print(predictions)

#predictionsLarge = list(dnnregressor.predict_scores(input_fn=eval_input_fn))
#print(predictionsLarge)

#predictionsLinear = list(regressor.predict_scores(input_fn=test_input_fn))
#print(predictionsLinear)

#predictionsLinearLarge = list(regressor.predict_scores(input_fn=eval_input_fn))
#print(predictionsLinearLarge)