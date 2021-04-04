#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


os.chdir(r'C:\Users\ITI Aundh 4\Deployment Heroku-01.04.2021')


# In[4]:


df = pd.read_csv('ai4i2020.csv')


# In[5]:


df.head()


# In[6]:


df.shape


# In[7]:


df.info()


# In[8]:


df.isna().sum()


# In[9]:


df.drop(['UDI','Product ID'],axis=1,inplace=True)


# In[10]:


df['Type'].unique()


# In[11]:


def type(x):
    if x=='L':
        return 1
    elif x=='M':
        return 2
    else:
        return 3
    
df['Type']=df['Type'].map(type)


# In[12]:


df.head()


# In[13]:


df.describe()


# ## Model Fitting:

# In[14]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import Ridge,Lasso,RidgeCV, LassoCV, ElasticNet, ElasticNetCV,LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor 
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.tree import DecisionTreeClassifier


# ### 2. Random Forest Classifier:

# In[15]:


from sklearn.ensemble import RandomForestClassifier


# In[16]:


df2=df.copy()


# In[17]:


df2.head()


# In[18]:


y=df2[['Machine failure','TWF', 'HDF', 'PWF', 'OSF', 'RNF']]
x=df2.drop(columns=['Machine failure','TWF', 'HDF', 'PWF', 'OSF', 'RNF'],axis=0)


# In[19]:


X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size = 0.30, random_state= 0)


# In[20]:


cls = RandomForestClassifier(n_estimators=100)
cls.fit(X_train, Y_train)


# In[21]:


acc_r=cls.score(X_train,Y_train)
acc_r


# In[22]:


acc_rfc=cls.score(X_test, Y_test)
acc_rfc


# In[23]:


y_pred=cls.predict(X_train)


# In[24]:


acc=accuracy_score(Y_train,y_pred)
acc


# In[25]:


y_pred=cls.predict(X_test)


# In[26]:


acc=accuracy_score(Y_test,y_pred)
acc


# In[27]:


auc2 = roc_auc_score(Y_test, y_pred)
auc2


# In[28]:


from sklearn.metrics import multilabel_confusion_matrix
print(multilabel_confusion_matrix(Y_test, y_pred))


# In[29]:


from sklearn import metrics
print(metrics.classification_report(Y_test,y_pred,digits=3))


# In[30]:


import pickle
pickle.dump(cls, open('randomforest.pkl','wb'))


# In[34]:


maintainance = pickle.load(open('randomforest.pkl','rb'))
result=maintainance.predict([[1,298.4,308.2,1282,60.7,216]])


# In[35]:


result


# In[36]:


if result[0][0]==1:
    print('Machine Failure')
    print('Reason:')
else:
    print('Machine OK')

if result[0][1]==1:
    print('Tool Wear Failure(TWF)')
if result[0][2]==1:
    print('Heat Dissipation Failure (HDF)')
if result[0][3]==1:
    print('Power Failure (PWF)')
if result[0][4]==1:
    print('Overstrain Failure (OSF)')
if result[0][5]==1:
    print('Random Failures (RNF)')


# In[ ]:




