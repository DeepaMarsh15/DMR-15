#!/usr/bin/env python
# coding: utf-8
##############Predicting Heart Disease####### PROJECT                       A) Data Analysis:
a. Import the dataset
b. Get information about dataset (mean, max, min, quartiles etc.)
c. Find the correlation between all fields.
# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix


# In[2]:


dataset=pd.read_csv('dataset.csv')


# In[3]:


dataset.head()


# In[4]:


dataset.describe()


# In[5]:


corr=dataset.corr()


# In[6]:


corr

B) Data Visualization:
a. Visualize the number of patients having a heart disease and not having a heart disease.
b. Visualize the age and weather patient has disease or not
c. Visualize correlation between all features using a heat map
# In[7]:


sns.countplot(dataset.target,palette=['green','red'])
plt.title('[0]:no heart disease [1]:have heart disease')
plt.show()


# In[8]:


plt.figure(figsize=(18,10))
sns.countplot(x='age',hue='target',data=dataset,palette=['green','red'])
plt.legend(['Does not have heart disease','Have heart disease'])
plt.title('Heart disease for age')
plt.xlabel('age')
plt.ylabel('frequency')
plt.plot()


# In[9]:


plt.figure(figsize=(18,10))
sns.heatmap(corr,annot=True)
plt.plot()

C) Logistic Regression:
a. Build a simple logistic regression model
i. Divide the dataset in 70:30 ratio
ii. Build the model on train set and predict the values on test set
iii. Build the confusion matrix and get the accuracy score
# In[10]:


x=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]


# In[11]:


x


# In[12]:


y


# In[13]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[14]:


x_train


# In[15]:


x_test


# In[16]:


clf_modl=LogisticRegression()


# In[17]:


clf_modl.fit(x_train,y_train)


# In[18]:


x_test


# In[20]:


y_pred=clf_modl.predict(x_test)


# In[21]:


y_pred


# In[22]:


y_test


# In[23]:


y_test=y_test.values
y_test


# In[24]:


y_pred


# In[25]:


log_score=accuracy_score(y_test,y_pred)


# In[26]:


log_score


# In[27]:


log_cm=confusion_matrix(y_test,y_pred)


# In[29]:


log_cm

D) Decision Tree:
a. Build a decision tree model
i. Divide the dataset in 70:30 ratio
ii. Build the model on train set and predict the values on test set
iii. Build the confusion matrix and calculate the accuracy
# In[30]:


dec_clf=DecisionTreeClassifier()


# In[31]:


dec_clf.fit(x_train,y_train)


# In[35]:


y_pred1=dec_clf.predict(x_test)


# In[36]:


y_pred1


# In[37]:


dec_score=accuracy_score(y_test,y_pred1)


# In[38]:


dec_score


# In[39]:


dec_cm=confusion_matrix(y_test,y_pred1)


# In[40]:


dec_cm

E) Random Forest:
a. Build a Random Forest model
i. Divide the dataset in 70:30 ratio
ii. Build the model on train set and predict the values on test set
iii. Build the confusion matrix and calculate the accuracy
# In[41]:


clf=RandomForestClassifier()


# In[42]:


clf.fit(x_train,y_train)


# In[43]:


y_pred2=clf.predict(x_test)


# In[44]:


y_pred2


# In[45]:


score_ac=accuracy_score(y_test,y_pred2)


# In[46]:


score_ac


# In[47]:


rf_cm=confusion_matrix(y_test,y_pred2)


# In[48]:


rf_cm

 Random Forest:0.8241758241758241
 Decision Tree:0.8021978021978022
 Logistic Regression:0.8351648351648352