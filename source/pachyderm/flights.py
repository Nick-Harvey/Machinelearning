import datetime, warnings, scipy 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import ConnectionPatch
from collections import OrderedDict
from matplotlib.gridspec import GridSpec
from mpl_toolkits.basemap import Basemap
from sklearn import metrics, linear_model
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from scipy.optimize import curve_fit
plt.rcParams["patch.force_edgecolor"] = True
plt.style.use('fivethirtyeight')
mpl.rc('patch', edgecolor = 'dimgray', linewidth=1)
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "last_expr"
pd.options.display.max_columns = 50

warnings.filterwarnings("ignore")

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
#df = pd.read_csv(path, header=0, skipinitialspace=True, names=names)
df = pd.read_csv(path)
print('Dataframe dimensions:', df.shape)

tab_info = pd.DataFrame(df.dtypes).T.rename(index={0:'column type'})
tab_info = tab_info.append(pd.DataFrame(df.isnull().sum()).T.rename(index={0:'null values (nb)'}))
tab_info = tab_info.append(pd.DataFrame(df.isnull().sum()/df.shape[0]*100).T.rename(index={0:'null values (%)'}))
tab_info

# Pull list of airports from airports.csv
airports = pd.read_csv("./airports.csv", header=1)

#plt airports and show number of flights
# count_flights = df['ORIGIN'].value_counts()
# #___________________________
# plt.figure(figsize=(11,11))
# #________________________________________
# # define properties of markers and labels
# colors = ['yellow', 'red', 'lightblue', 'purple', 'green', 'orange']
# size_limits = [1, 100, 1000, 10000, 100000, 1000000]
# labels = []
# for i in range(len(size_limits)-1):
#     labels.append("{} <.< {}".format(size_limits[i], size_limits[i+1])) 
# #____________________________________________________________
# map = Basemap(resolution='i',llcrnrlon=-180, urcrnrlon=-50,
#               llcrnrlat=10, urcrnrlat=75, lat_0=0, lon_0=0,)
# map.shadedrelief()
# map.drawcoastlines()
# map.drawcountries(linewidth = 3)
# map.drawstates(color='0.3')

# for index, (code, y,x) in airports[['IATA_CODE', 'LATITUDE', 'LONGITUDE']].iterrows():
#     try:
#         x, y = map(x, y)
#         isize = [i for i, val in enumerate(size_limits) if val < count_flights[code]]
#         ind = isize[-1]
#         map.plot(x, y, marker='o', markersize = ind+5, markeredgewidth = 1, color = colors[ind],
#             markeredgecolor='k', label = labels[ind])
#     except KeyError:
#         pass


# print(df.index)
# print(df.columns)

# handles, labels = plt.gca().get_legend_handles_labels()
# by_label = OrderedDict(zip(labels, handles))
# key_order = ('1 <.< 100', '100 <.< 1000', '1000 <.< 10000',
#              '10000 <.< 100000', '100000 <.< 1000000')
# new_label = OrderedDict()
# for key in key_order:
#     try:
#         new_label[key] = by_label[key]
#     except KeyError:
#         pass
# plt.legend(new_label.values(), new_label.keys(), loc = 1, prop= {'size':11},
#            title='Number of flights per year', frameon = True, framealpha = 1)
# plt.show()

#
#
# Clean all the data
#
#

# Create Date 
#df.rename(columns={'DAY_OF_MONTH': 'DAY'}, inplace=True)
df['DAY'] = df['DAY_OF_MONTH']
df['DATE'] = pd.to_datetime(df[['YEAR','MONTH','DAY']])

print("Success")

#df['DATE'] = pd.DataFrame({'year': ['YEAR'],'month': ['MONTH'],'day': ['DAY_OF_MONTH']})

#_________________________________________________________
# Function that convert the 'HHMM' string to datetime.time
def format_heure(chaine):
    if pd.isnull(chaine):
        return np.nan
    else:
        if chaine == 2400: chaine = 0
        chaine = "{0:04d}".format(int(chaine))
        heure = datetime.time(int(chaine[0:2]), int(chaine[2:4]))
        return heure
#_____________________________________________________________________
# Function that combines a date and time to produce a datetime.datetime
def combine_date_heure(x):
    if pd.isnull(x[0]) or pd.isnull(x[1]):
        return np.nan
    else:
        return datetime.datetime.combine(x[0],x[1])
#_______________________________________________________________________________
# Function that combine two columns of the dataframe to create a datetime format
def create_flight_time(df, col):    
    liste = []
    for index, cols in df[['DATE', col]].iterrows():    
        if pd.isnull(cols[1]):
            liste.append(np.nan)
        elif float(cols[1]) == 2400:
            cols[0] += datetime.timedelta(days=1)
            cols[1] = datetime.time(0,0)
            liste.append(combine_date_heure(cols))
        else:
            cols[1] = format_heure(cols[1])
            liste.append(combine_date_heure(cols))
    return pd.Series(liste)

# Modify dataframe variables
df['CRS_DEP_TIME'] = create_flight_time(df, 'CRS_DEP_TIME')
df['DEP_TIME'] = df['DEP_TIME'].apply(format_heure)
df['CRS_ARR_TIME'] = df['CRS_ARR_TIME'].apply(format_heure)
df['ARR_TIME'] = df['ARR_TIME'].apply(format_heure)
#__________________________________________________________________________
df.loc[:5, ['CRS_DEP_TIME', 'CRS_ARR_TIME', 'DEP_TIME',
             'ARR_TIME', 'DEP_DELAY', 'ARR_DELAY']]


variables_to_remove = ['CANCELLED','DIVERTED','CANCELLATION_CODE']
df.drop(variables_to_remove, axis = 1, inplace = True)
df = df[['CARRIER', 'ORIGIN', 'DEST',
        'CRS_DEP_TIME', 'DEP_TIME', 'DEP_DELAY',
        'CRS_ARR_TIME', 'ARR_TIME', 'ARR_DELAY',
        'DISTANCE']]
df[:5]

missing_df = df.isnull().sum(axis=0).reset_index()
missing_df.columns = ['variable', 'missing values']
missing_df['filling factor (%)']=(df.shape[0]-missing_df['missing values'])/df.shape[0]*100
missing_df.sort_values('filling factor (%)').reset_index(drop = True)

df.dropna(inplace = True)


# Comparing Airlines
airlines_names = pd.read_csv('./carriers.csv')

# I extract a subset of columns and redefine the airlines labeling 
abbr_companies = airlines_names.set_index('IATA_CODE')['AIRLINE'].to_dict()
df2 = df.loc[:, ['CARRIER', 'DEP_DELAY']]
df2['AIRLINE'] = df2['CARRIER'].replace(abbr_companies)

abbr_companies = airlines_names.set_index('IATA_CODE')['AIRLINE'].to_dict()

def get_stats(group):
    return {'min': group.min(), 'max': group.max(),
            'count': group.count(), 'mean': group.mean()}
#_______________________________________________________________
# Creation of a dataframe with statitical infos on each airline:
global_stats = df['DEP_DELAY'].groupby(df['CARRIER']).apply(get_stats).unstack()
global_stats = global_stats.sort_values('count')
global_stats



#---------------------------------------------------------------#
#                                                               #
#               Plot out the delays per airline                 #
#                                                               #
#---------------------------------------------------------------#

#___________________________________________
# Model function used to fit the histograms
def func(x, a, b):
    return a * np.exp(-x/b)
#-------------------------------------------
points = [] ; label_company = []
fig = plt.figure(1, figsize=(11,11))
i = 0
for carrier_name in [abbr_companies[x] for x in global_stats.index]:
    i += 1
    ax = fig.add_subplot(5,3,i)    
    #_________________________
    # Fit of the distribution
    n, bins, patches = plt.hist(x = df2[df2['AIRLINE']==carrier_name]['DEP_DELAY'],
                                range = (15,180), normed=True, bins= 60)
    bin_centers = bins[:-1] + 0.5 * (bins[1:] - bins[:-1])    
    popt, pcov = curve_fit(func, bin_centers, n, p0 = [1, 2])
    #___________________________
    # bookeeping of the results
    points.append(popt)
    label_company.append(carrier_name)
    #______________________
    # draw the fit curve
    plt.plot(bin_centers, func(bin_centers, *popt), 'r-', linewidth=3)    
    #_____________________________________
    # define tick labels for each subplot
    if i < 10:
        ax.set_xticklabels(['' for x in ax.get_xticks()])
    else:
        ax.set_xticklabels(['{:2.0f}h{:2.0f}m'.format(*[int(y) for y in divmod(x,60)])
                            for x in ax.get_xticks()])
    #______________
    # subplot title
    plt.title(carrier_name, fontsize = 14, fontweight = 'bold', color = 'darkblue')
    #____________
    # axes labels 
    if i == 4:
        ax.text(-0.3,0.9,'Normalized count of flights', fontsize=16, rotation=90,
            color='k', horizontalalignment='center', transform = ax.transAxes)
    if i == 14:
        ax.text( 0.5, -0.5 ,'Delay at origin', fontsize=16, rotation=0,
            color='k', horizontalalignment='center', transform = ax.transAxes)
    #___________________________________________
    # Legend: values of the a and b coefficients
    ax.text(0.68, 0.7, 'a = {}\nb = {}'.format(round(popt[0],2), round(popt[1],1)),
            style='italic', transform=ax.transAxes, fontsize = 12, family='fantasy',
            bbox={'facecolor':'tomato', 'alpha':0.8, 'pad':5})
    
plt.tight_layout()


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
#training_label = df['DEST_AIRPORT_ID']
#eval_label = eval_data.pop('FL_NUM')
#test_label = test_data.pop('FL_NUM')

training_input_fn = tf.estimator.inputs.pandas_input_fn(x=training_data, y=training_label, batch_size=0, shuffle=False)

#eval_input_fn = tf.estimator.inputs.pandas_input_fn(x=eval_data, y=eval_label, batch_size=64, shuffle=False)

#test_input_fn = tf.estimator.inputs.pandas_input_fn(x=test_data, y=test_label, batch_size=10, shuffle=False)

#Feature columns
#carrier = tf.feature_column.categorical_column_with_vocabulary_list('CARRIER', vocabulary_list=['WN', 'OO', 'NK', 'AA', 'DL', 'UA'])
#distance = tf.feature_column.numeric_column('DISTANCE')
#dep_time = tf.feature_column.numeric_column('DEP_TIME')
#taxi_out = tf.feature_column.numeric_column('TAXI_OUT')
#fl_num = tf.feature_column.numeric_column('FL_NUM')
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