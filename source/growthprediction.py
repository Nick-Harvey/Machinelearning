import pandas as pd
import numpy as np
from fbprophet import Prophet
import matplotlib.pyplot as plt

#df = pd.read_csv('~/tmp/OpenShiftOps2.csv')
df = pd.read_csv('~/tmp/OpenShiftRevenue3.csv')
m = Prophet().fit(df)

future = m.make_future_dataframe(periods=120, freq='M')
fcst = m.predict(future)
fcst[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

m.plot(fcst);
m.plot_components(fcst);
plt.show()


# df.head()

# m = Prophet()
# m.fit(df);

# #future = m.make_future_dataframe(periods=365)
# future = m.make_future_dataframe(periods=60, freq='M')
# future.tail()

# forecast = m.predict(future)
# forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

# m.plot(forecast);

# m.plot_components(forecast);

# plt.show()
