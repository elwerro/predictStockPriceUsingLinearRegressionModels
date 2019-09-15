#
#from github avinash007
#https://github.com/Avinash007/machine-learning/blob/master/stocks-regression/
#also find this script in my github
#https://github.com/elwerro/predictStockPriceUsingLinearRegressionModels

#!pip install Quandl

#to get current dates
# import datetime as dt

#to plot/visualize data
import matplotlib.pyplot as plt
from matplotlib import style

#Seaborn is a Python data visualization library based on matplotlib. 
#It provides a high-level interface for drawing attractive and informative statistical graphics.
import seaborn as sns

#to get data from www.quandl.com
import quandl

#data structure for data 
import pandas as pd

#to do math on python data structures
import numpy as np



style.use('ggplot')
start_date = '2017-1-1'
end_date = '2019-1-31'
quandl.ApiConfig.api_key = 'wRJYSd36h3b_9FngYFER'


stock_data = quandl.get('EOD/V', start_date=start_date, end_date=end_date)

print(stock_data)

print(stock_data.columns)

stock_data = stock_data[['Open', 'Close', 'High', 'Low', 'Volume', 'Adj_Close']]
print(stock_data.head())

print(stock_data.tail())

print(stock_data.info())

print(stock_data.describe())

fig = plt.figure(1)
corr = stock_data.corr()
fig.canvas.set_window_title('Heatmap') 
sns.heatmap(corr, annot=True)

df_test = stock_data[-20:]
print(df_test.shape)
print(df_test.head())

df_train = stock_data[:-20]
print(df_train.shape)


# Plotting Train and Test data
fig = plt.figure(2, figsize=[10,8])
ax = plt.subplot(111)
ax.plot(df_train['Adj_Close'], label='Train')
ax.plot(df_test['Adj_Close'], label='Valid')
ax.legend()
fig.canvas.set_window_title('Visa stock 2017-2018') 
# plt.show()


# We will use the last 6 day data to make prediction
window = 7
train_data = df_train['Adj_Close']
test_data = df_test['Adj_Close']
index = len(train_data) - window
print(index)

data = pd.DataFrame(np.zeros((index, window)))
for row in range(index):
	for col in range(window):
		data.iloc[row,col] = train_data[col+row]
print(data)

y_train = data.iloc[:,-1]
print(y_train.shape)

X_train = data.iloc[:,:-1]
print(X_train.shape)

index = len(test_data) - window

data = pd.DataFrame(np.zeros((index, window)))

for row in range(index):
	for col in range(window):
		data.iloc[row,col] = train_data[col+row]


y_test = data.iloc[:,-1]
print(y_test.shape)


X_test = data.iloc[:,:-1]
print(X_test.shape)

#Model Building

# Linear Regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred_lr)
print("Root Mean Squared Error: ", np.sqrt(mse))

fig = plt.figure(3, figsize=[10,8])
ax = plt.subplot(111)
ax.plot(y_test.index, y_pred_lr, label='Predicted')
ax.plot(y_test, label='Test')
ax.legend()
fig.suptitle("Root Mean Squared Error: {}".format(np.sqrt(mse)))
fig.canvas.set_window_title('Linear Regression Model') 
# plt.show()

# XGBoost
import xgboost as xgb

xgb_model = xgb.XGBRegressor(colsample_bytree=0.4,
			                 gamma=0,
			                 learning_rate=0.07,
			                 max_depth=3,
			                 min_child_weight=1.5,
			                 n_estimators=10000,
			                 reg_alpha=0.75,
			                 reg_lambda=0.45,
			                 subsample=0.6,
			                 seed=42)

xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred_xgb)
print("Root Mean Squared Error: ", np.sqrt(mse))

fig = plt.figure(4, figsize=[10,8])
ax = plt.subplot(111)
ax.plot(y_test.index, y_pred_xgb, label='Predicted')
ax.plot(y_test, label='Test')
ax.legend()
fig.suptitle("Root Mean Squared Error: {}".format(np.sqrt(mse)))
fig.canvas.set_window_title('XGBoost Model') 
# plt.show()

# Keras LSTM
from sklearn.linear_model import Ridge

ridge_model = Ridge()
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred_ridge)
print("Root Mean Squared Error: ", np.sqrt(mse))

fig = plt.figure(5, figsize=[10,8])
ax = plt.subplot(111)
ax.plot(y_test.index, y_pred_ridge, label='Predicted')
ax.plot(y_test, label='Test')
ax.legend()
fig.suptitle("Root Mean Squared Error: {}".format(np.sqrt(mse)))
fig.canvas.set_window_title('Ridge Regression Model')
plt.show()




