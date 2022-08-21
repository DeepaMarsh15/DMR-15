#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install plotly')


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# In[3]:


import warnings
warnings.filterwarnings('ignore')


# In[4]:


df=pd.read_csv('census-income.csv')


# In[5]:


df.head(15)


# In[6]:


df.shape


# In[7]:


df.dtypes


# In[8]:


df.isnull().sum()


# In[9]:


df.nunique()


# In[10]:


df.describe()

1. Data Preprocessing:
a) Replace all the missing values with NA.
# In[11]:


df['age']=df['age'].replace('?','NA')
df['workclass']=df['workclass'].replace('?','NA')
df['fnlwgt']=df['fnlwgt'].replace('?','NA')
df['education']=df['education'].replace('?','NA')
df['education-num']=df['education-num'].replace('?','NA')
df['marital-status']=df['marital-status'].replace('?','NA')
df['occupation']=df['occupation'].replace('?','NA')

b) Remove all the rows that contain NA values. 
# In[12]:


df.dropna()


# In[13]:


df['income'].value_counts().plot(kind='bar')


# In[14]:


print(df['income'].value_counts())
df['income'].value_counts().plot.pie(autopct='%1.1f%%')


# In[15]:


df.income = df.income.replace('<=50K',0)
df.income = df.income.replace('>=50K',1)


# In[16]:


df.corr()


# In[17]:


sns.heatmap(df.corr(), annot=True);


# In[18]:


df.hist(figsize=(12,12),layout=(3,3), sharex= False);

a) Extract the “education” column and store it in “census_ed” 
# In[19]:


df.rename(columns={"education":"census_ed"},inplace=True) 

b)Extract all the columns from “age” to “relationship” and store it in “census_seq”.
# In[104]:


census_seq=df.columns(:(0:8))

c)Extract the column number “5”, “8”, “11” and store it in “census_col”.
# In[105]:


census_col=df{('marital-status','race','capital-loss')}

d) Extract all the male employees who work in state-gov and store it in “male_gov”.
# In[106]:


male_gov=df['occupation','state-gov','Male'=True]

e) Extract all the 39 year olds who either have a bachelor's degree or who are native of the United States and store the result in “census_us”.f) Extract 200 random rows from the “census” data frame and store it in “census_200”.
# In[108]:


from random import random
   
lst = []
  
for i in range(200):
  lst.append(random())
    
# Prints random items
print(lst)

g) Get the count of different levels of the “workclass” column.
# In[110]:


df["workclass"].value_counts()

h) Calculate the mean of the “capital.gain” column grouped according to “workclass”.i) Create a separate dataframe with the details of males and females from the census data that has income more than 50,000. j) Calculate the percentage of people from the United States who are private employees and earn less than 50,000 annually. k) Calculate the percentage of married people in the census data.l) Calculate the percentage of high school graduates earning more than 50,000 annually. 3. Linear Regression:
●	Divide the dataset into training and test sets in 70:30 ratio.
●	Build a linear model on the test set where the dependent variable is “hours.per.week” and the independent variable is “education.num”.
●	Predict the values on the train set and find the error in prediction. 
●	Find the root-mean-square error (RMSE)
# In[22]:


x=df.iloc[:,:-1]
y=df.iloc[:,-1]


# In[23]:


from sklearn.preprocessing import StandardScaler,LabelEncoder


# In[24]:


df=df.apply(LabelEncoder().fit_transform)
df.head()


# In[25]:


sc=StandardScaler().fit(df.drop('income',axis=1))


# In[26]:


x=sc.transform(df.drop('income',axis=1))
y=df['income']


# In[27]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)


# In[100]:


x_train


# In[28]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_test,y_test)


# In[29]:


x_pred=regressor.predict(x_train)


# In[30]:


from sklearn.metrics import accuracy_score
print('Accuracy on training data:{:,.3f}'.format(regressor.score(x_train,y_train)))
print('Accuracy on test data:{:,.3f}'.format(regressor.score(x_test,y_test)))


# In[102]:


from sklearn.metrics import mean_squared_error
import math
from math import sqrt
rmse=math.sqrt(mean_squared_error(x_train, x_pred))
print(rmse)

4. Logistic Regression:
 a) Build a simple logistic regression model as follows:
●Divide the dataset into training and test sets in 65:35 ratio.
●Build a logistic regression model where the dependent variable is “X”(yearly income) and the independent variable is “occupation”.
●Predict the values on the test set.
●Build a confusion matrix and find the accuracy.

# In[32]:


X=df.iloc[:,-1]
Y=df.iloc[:,-9]


# In[33]:


X


# In[34]:


Y


# In[35]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.35,random_state=1)


# In[36]:


X_train


# In[37]:


X_test


# In[38]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()


# In[39]:


lr.fit(X_train.values.reshape(-1,1),Y_train)


# In[41]:


Y_pred=lr.predict(X_test)


# In[ ]:




b)Build a multiple logistic regression model as follows:
●Divide the dataset into training and test sets in 80:20 ratio.
●Build a logistic regression model where the dependent variable is “X”(yearly income) and independent variables are “age”, “workclass”, and “education”.
●Predict the values on the test set.
●Build a confusion matrix and find the accuracy
# In[42]:


X1=df.iloc[:,-1]
Y1=df.iloc[:,0]


# In[43]:


X1_train,X1_test,Y1_train,Y1_test=train_test_split(X1,Y1,test_size=0.2,random_state=1)


# In[44]:


from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()


# In[45]:


X1_train


# In[46]:


X1_test


# In[47]:


model=LogisticRegression()


# In[48]:


model.fit(X1_train.values.reshape(-1,1),Y1_train)


# In[49]:


X1_test


# In[50]:


Y1_pred=model.predict(X1_test)

5. Decision Tree:
a) Build a decision tree model as follows:

●Divide the dataset into training and test sets in 70:30 ratio.
●Build a decision tree model where the dependent variable is “X”(Yearly Income) and the rest of the variables as independent variables.
●Predict the values on the test set.
●Build a confusion matrix and calculate the accuracy.
# In[51]:


from sklearn.tree import DecisionTreeClassifier


# In[52]:


dec_clf=DecisionTreeClassifier()


# In[53]:


x1=df.iloc[:,-1]
y1=df.iloc[:,:-1]


# In[54]:


x1


# In[55]:


y1


# In[56]:


x1_train,x1_test,y1_train,y1_test=train_test_split(x1,y1,test_size=0.3)


# In[57]:


dec_clf.fit(x1_train.values.reshape(-1,1),y1_train)


# In[ ]:




6. Random Forest:
a) Build a random forest model as follows:
●Divide the dataset into training and test sets in 80:20 ratio.
●Build a random forest model where the dependent variable is “X”(Yearly Income) and the rest of the variables as independent variables and number of trees as 300.
●Predict values on the test set
●Build a confusion matrix and calculate the accuracy
# In[58]:


from sklearn.ensemble import RandomForestClassifier


# In[59]:


clf=RandomForestClassifier()


# In[60]:


X2=df.iloc[:,-1]
Y2=df.iloc[:,:-1]


# In[61]:


X2


# In[62]:


Y2


# In[63]:


X2_train,X2_test,Y2_train,Y2_test=train_test_split(X2,Y2,test_size=0.2)


# In[64]:


clf.fit(X2_train.values.reshape(-1,1),Y2_train)


# In[ ]:




7. For this problem, use the population dataset, and perform the following:
1.EDA on the time series to find trends and seasonality.
2.Forecast the population on the given dataset for the next 6 months. 
# In[75]:


DF=pd.read_csv('popdata.csv', parse_dates=["date"])


# In[76]:


DF.head()


# In[77]:


DF.shape


# In[89]:


DF.nunique()


# In[90]:


DF.isnull().sum()


# In[93]:


DF.describe()


# In[78]:


import plotly
import plotly.graph_objects as go


# In[79]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[92]:


DF.plot(kind='scatter', x='date', y='value');


# In[83]:


total_value=DF.groupby('date').sum().reset_index()


# In[88]:


def plot_DF(DF, x, y, title="", xlabel='date', ylabel='value', dpi=100)
    plt.figure(figsize=(16,5), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()
    plot_DF(DF, x=DF.index, y=DF.value, title='Value as per date')    


# In[ ]:




