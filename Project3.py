#!/usr/bin/env python
# coding: utf-8

# #### Importing required libary and data

# In[44]:


import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score


# In[45]:


os.chdir(r'C:\Users\ITI Aundh 4\Deployment Heroku-01.04.2021')


# In[46]:


df = pd.read_csv('ai4i2020.csv')


# In[47]:


df.head()


# In[48]:


df.shape


# In[49]:


df.info()


# In[50]:


df.isna().sum()


# In[51]:


df.drop(['UDI','Product ID'],axis=1,inplace=True)


# In[52]:


df['Type'].unique()


# In[53]:


def type(x):
    if x=='L':
        return 1
    elif x=='M':
        return 2
    else:
        return 3
    
df['Type']=df['Type'].map(type)


# In[54]:


df.head()


# In[55]:


df.describe()


# In[56]:


df.columns

df.drop('Torque [Nm]',axis=1,inplace=True)

# In[57]:


y=df["Machine failure"]
x=df.drop(columns=["Machine failure"])


# In[58]:


X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=0)


# In[59]:


clf=DecisionTreeClassifier()
clf.fit(X_train,Y_train)


# In[60]:


y_pred=clf.predict(X_train)


# In[61]:


acc=accuracy_score(Y_train,y_pred)
acc


# In[62]:


y_pred=clf.predict(X_test)


# In[63]:


acc=accuracy_score(Y_test,y_pred)
acc


# In[64]:


auc = roc_auc_score(Y_test, y_pred)
auc


# In[65]:


grid_param = {
    'criterion': ['gini', 'entropy'],
    'max_depth' : range(2,32,1),
    'min_samples_leaf' : range(1,10,1),
    'min_samples_split': range(2,10,1),
    'splitter' : ['best', 'random']
    
}


# In[66]:


grid_search = GridSearchCV(estimator=clf,
                     param_grid=grid_param,
                     cv=5,
                    n_jobs =-1)


# In[67]:


grid_search.fit(X_train,Y_train)


# In[68]:


best_parameters = grid_search.best_params_
print(best_parameters)


# In[69]:


grid_search.best_score_


# In[70]:


clf = DecisionTreeClassifier(criterion = 'gini', max_depth =4, min_samples_leaf= 1, min_samples_split= 2, splitter ='best')
clf.fit(X_train,Y_train)


# In[71]:


clf.score(X_train,Y_train)


# In[72]:


clf.score(X_test,Y_test)


# In[73]:


y_pred=clf.predict(X_train)


# In[74]:


acc=accuracy_score(Y_train,y_pred)
acc


# In[75]:


y_pred=clf.predict(X_test)


# In[76]:


acc=accuracy_score(Y_test,y_pred)
acc


# In[77]:


auc = roc_auc_score(Y_test, y_pred)
auc


# **ROC**

# In[78]:


fpr, tpr, thresholds = roc_curve(Y_test, y_pred)


# In[79]:


plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--',label='ROC curve (area = %0.2f)' % auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()


# In[80]:


import pickle
pickle.dump(clf, open('Project3.pkl','wb'))

