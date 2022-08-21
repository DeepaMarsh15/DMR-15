#!/usr/bin/env python
# coding: utf-8
                        Decision Tree AssignmentYou work in XYZ Company as a Python Data Scientist. The company officials have collected some data
on salaries based on year of experience and wish for you to create a model from it.
Dataset: diabetes.csv

Tasks to be performed:
1. Load the dataset using pandas
2. Extract data fromOutcome column is a variable named Y
3. Extract data from every column except Outcome column in a variable named X
4. Divide the dataset into two parts for training and testing in 70% and 30% proportion
5. Create and train Decision Tree Model on training set
6. Make predictions based on the testing set using the trained model
7. Check the performance by calculating the confusion matrix and accuracy score of the mode
# In[3]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

OR
from sklearn import *----------to import everything together

1. Load the dataset using pandas
# In[4]:


dataset=pd.read_csv('diabetes.csv')


# In[6]:


dataset.head()

2. Extract data fromOutcome column is a variable named Y
# In[7]:


Y=dataset.iloc[:,-1]


# In[8]:


Y

3. Extract data from every column except Outcome column in a variable named X
# In[9]:


X=dataset.iloc[:,:-1]


# In[10]:


X

4. Divide the dataset into two parts for training and testing in 70% and 30% proportion
# In[11]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.3)


# In[13]:


print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

5. Create and train Decision Tree Model on training set
# In[14]:


clf=DecisionTreeClassifier()


# In[15]:


clf.fit(X_train,Y_train)

6. Make predictions based on the testing set using the trained model 
# In[16]:


y_pred=clf.predict(X_test)


# In[17]:


y_pred

7. Check the performance by calculating the confusion matrix and accuracy score of the model
# In[19]:


accuracy_score(Y_test,y_pred)

                      Random Forest AssignmentProblem Statement:
You work in XYZ Company as a Python Data Scientist. The company officials have collected some data
on salaries based on year of experience and wish for you to create a model from it.
Dataset: diabetes.csv
Tasks to be performed:
1. Load the dataset using pandas
2. Extract data fromOutcome column is a variable named Y
3. Extract data from every column except Outcome column in a variable named X.
4. Divide the dataset into two parts for training and testing in 70% and 30% proportion
5. Create and train Random Forest Model on training set
6. Make predictions based on the testing set using the trained model
7. Check the performance by calculating the confusion matrix and accuracy score of the model Till Q.4 it is same as above(Decision Tree)5. Create and train Random Forest Model on training set
# In[20]:


clf=RandomForestClassifier()


# In[21]:


clf.fit(X_train,Y_train)

6. Make predictions based on the testing set using the trained model
# In[22]:


Ry_pred=clf.predict(X_test)


# In[23]:


Ry_pred

7. Check the performance by calculating the confusion matrix and accuracy score of the model
# In[24]:


accuracy_score(Y_test,Ry_pred)


# In[ ]:




