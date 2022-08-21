#!/usr/bin/env python
# coding: utf-8
################ ARIMA  ASSIGNMENT #####################Q1.Load the dataset using Pandas Library along with needed formatting.a.Check and remove the null values b.Plot the needed feature
# In[1]:


get_ipython().system('pip install pmdarima')


# In[2]:


import pandas as pd
import numpy as np


# In[3]:


df=pd.read_csv('MaunaLoaDailyTemps.csv',index_col='DATE',parse_dates=True)
df=df.dropna()
print('shape=',df.shape)
df.head()


# In[4]:


df['AvgTemp'].plot(figsize=(12,5))

Q.2 Build a function to check whether the given model is stationary or not
# In[5]:


from statsmodels.tsa.stattools import adfuller


# In[6]:


def adf_test(dataset):
    dftest=adfuller(dataset,autolag='AIC')
    print('1.ADF:',dftest[0])
    print('2.p-value',dftest[1])
    print('3.number of lags',dftest[2])
    print('4.number of observations used for ADF Regression and critical value calculation',dftest[3])
    print('5.critical values')
    
    for key, val in dftest[4],items():
        print('\t',key,':',val) 
    adf_test(df['AvgTemp'])


# In[7]:


adf_test(df['AvgTemp']) 

Q.3 Load the Required  Package used for implementing ARIMA.(Hint:pmdarima)
# In[8]:


from pmdarima import auto_arima
import warnings
warnings.filterwarnings('ignore')

Q.4 Identify the best suit Model along with the orders
# In[9]:


stepwise_fit=auto_arima(df['AvgTemp'],supress_warnings=True)
stepwise_fit.summary()

Q.5 Train Model
a.Split the Data in Training and Testing Datasets
b.Fit the Model in top up Algorithm
c.Perform Time Series forecasting with the help of Test dataset
d.PLot the Prediction along with Actual Values
# In[10]:


from statsmodels.tsa.arima_model import ARIMA


# In[11]:


print(df.shape)


# In[12]:


train=df.iloc[:-30]


# In[13]:


test=df.iloc[-30:]


# In[14]:


print(train.shape,test.shape)


# In[15]:


print(test.iloc[0],test.iloc[-1])


# In[16]:


model=ARIMA(train['AvgTemp'],order=(1, 0, 5))
model=model.fit()


# In[17]:


model.summary()


# In[18]:


start=len(train)


# In[19]:


end=len(train)+len(test)-1


# In[20]:


index_future_dates=pd.date_range(start='2018-12-01',end='2018-12-30')


# In[21]:


pred=model.predict(start=start,end=end).rename('ARIMA predictions')


# In[22]:


pred.index=index_future_dates
pred.plot(legend=True)
test['AvgTemp'].plot(legend=True)

Q.6 Check the RMSE to compare the Efficacy of the Model
# In[23]:


from sklearn.metrics import mean_squared_error
from math import sqrt
rmse=sqrt(mean_squared_error(pred,test['AvgTemp']))
print(rmse)


# In[24]:


model2=ARIMA(df['AvgTemp'],order=(1,0,5))
model2=model2.fit()
df.tail()

Q.7 Perform Time Series Forecasting and try to forecast the Average Temperature for the next 30 days
# In[28]:


index_future_dates=pd.date_range(start='2018-12-30',end='2019-01-29')
#print(index_future_dates)
pred=model2.predict(start=len(df),end=len(df)+30,typ='levels').rename('ARIMAPrediction')
#print(comp_pred)
pred.index=index_future_dates
print(pred)


# In[29]:


pred.plot(figsize=(12,15),legend=True)

