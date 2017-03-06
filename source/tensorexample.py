import tempfile
import urllib.request
import pandas as pd
import tensorflow as tf

#data and data prep
train_file = tempfile.NamedTemporaryFile()
test_file = tempfile.NamedTemporaryFile()
urllib.request.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", train_file.name)
urllib.request.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", test_file.name)

COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation", "relationship", "race", "gender", "capital_gain", "capital_loss", "hours_per_week", "native_country", "income_bracket"]

df_train = pd.read_csv(train_file, names=COLUMNS, skipinitialspace=True)
df_test = pd.read_csv(test_file, names=COLUMNS, skipinitialspace=True, skiprows=1)

LABEL_COLUMN = "label"
df_train[LABEL_COLUMN] = (df_train["income_bracket"].apply(lambda x: ">50k" in x)).astype(int)
df_test[LABEL_COLUMN] = (df_train["income_bracket"].apply(lambda x: ">50k" in x)).astype(int)

CATEGORICAL_COLUMNS = ["workclass", "education", "marital_staus", "occupation", "relationship", "race", "gender", "native_country"]
CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]


## Tensorflow starts here

def input_fn(df):
    # creates a dictionary mapping from each continous feature column name (k) to the values
    # of that column stored in a contstant Tensor.
    continuous_cols = {k: tf.constant(df[k].values)
            for k in CONTINUOUS_COLUMNS}

    # creates a dictionary mapping from each categorical feature column name (k) to the values
    # of that column stored in a TF.SparseTensor
    categorical_cols = {k: tf.SparseTensor(
        indices=[[i, 0] for i in range(df[k].size)],
        values=df[k].values,
        shape=[df[k].size, 1])
        for k in CATEGORICAL_COLUMNS}

    print(continuous_cols).items
    print(categorical_cols).items

    # Merges the two directories
    feature_cols = dict(continuous_cols.items() + categorical_cols.items())

    # Converts the label column into a constant Tensor
    label = tf.constant(df[LABEL_COLUMN].values)

    # Returns the feature column and the label
    return feature_cols, label

def train_input_fn():
    return input_fn(df_train)

def eval_input_fn():
    return input_fn(df_test)

# Base categorical features
gender = tf.contrib.layers.sparse_column_with_keys(column_name="gender", keys=["Female", "Male"])
education = tf.contrib.layers.sparse_column_with_hash_bucket("education", hash_bucket_size=1000)
relationship = tf.contrib.layers.sparse_column_with_hash_bucket("relationship", hash_bucket_size=100)
workclass = tf.contrib.layers.sparse_column_with_hash_bucket("workclass", hash_bucket_size=100)
occupation = tf.contrib.layers.sparse_column_with_hash_bucket("occupation", hash_bucket_size=1000)
#marital_status = tf.contrib.layers.sparse_column_with_hash_bucket("marital_status", hash_bucket_size=1000)
native_country = tf.contrib.layers.sparse_column_with_hash_bucket("native_country", hash_bucket_size=1000)

# Base continuous Feature Columns
age = tf.contrib.layers.real_valued_column("age")
education_num = tf.contrib.layers.real_valued_column("education_num")
capital_gain = tf.contrib.layers.real_valued_column("capital_gain")
capital_loss = tf.contrib.layers.real_valued_column("capital_loss")
hours_per_week = tf.contrib.layers.real_valued_column("hours_per_week")

# Buckets
age_buckets = tf.contrib.layers.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

# Intersecting columns
education_x_occupation = tf.contrib.layers.crossed_column([education, occupation], hash_bucket_size=int(1e4))

age_buckets_x_education_x_occupation = tf.contrib.layers.crossed_column([age_buckets, education, occupation], hash_bucket_size=int(1e6))

model_dir = tempfile.mkdtemp()
m = tf.contrib.learn.LinearClassifier(feature_columns=[
  gender, native_country, education, occupation, workclass,
  age_buckets, education_x_occupation, age_buckets_x_education_x_occupation],
  model_dir=model_dir)

# Train and fit
m.fit(input_fn=train_input_fn, steps=200)

results = m.evaluate(input_fn=eval_input_fn, steps=1)
for key in sorted(results):
    print(key, results[key])
