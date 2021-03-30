#!/usr/bin/env python
# coding: utf-8

# #### Importing required libary and data

# In[61]:


import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[62]:


os.chdir(r'F:\Mayur Ghate\Python Programing\Python by Uday Sir\Assignment\POC Project')


# In[63]:


df = pd.read_csv('ai4i2020.csv')


# In[64]:


df.head()


# ### 1) Data Pre Processing:

# In[65]:


df.shape


# #### Data has 10,000 rows and 14 columns

# In[66]:


df.info()


# #### Checking is their any null values in data set.

# In[67]:


df.isna().sum()


# #### Dropping UDI & Product ID

# In[68]:


df.drop(['UDI','Product ID'],axis=1,inplace=True)


# #### One Hot Encoding

# In[69]:


df['Type'].unique()


# In[70]:


def type(x):
    if x=='L':
        return 1
    elif x=='M':
        return 2
    else:
        return 3
    
df['Type']=df['Type'].map(type)


# In[71]:


df.head()


# In[72]:


df.describe()


# ### 2) Data Visualization:

# In[73]:


sns.distplot(df)


# In[74]:


df.columns


# In[75]:


df.boxplot(figsize=(18,10))


# #### This box plot shows that their is outliers present in data set. 'Rotational speed [rpm]' contains more outliers.

# In[76]:


df['Rotational speed [rpm]'].plot(kind='box')


# In[77]:


df[['TWF', 'HDF', 'PWF', 'OSF', 'RNF']].sum().sort_values(ascending=False).plot(kind='bar')
df[['TWF', 'HDF', 'PWF', 'OSF', 'RNF']].sum().sort_values(ascending=False)


# #### The machine failure consists of five independent failure modes. If at least one of the above failure modes is true, the process fails and the 'machine failure' label is set to 1. 
# Above Graph shows that -
# 1. 'heat dissipation failure'(HDF) has highest value i.e., machine failure occure most of the time due to HDF.
# 2. 'random failures'(RNF) has lowest value.

# In[78]:


df[['TWF', 'HDF', 'PWF', 'OSF', 'RNF']].sum().sum()


# ### 3) Model Fitting:

# ## 1. Logistic Regression Model:

# In[79]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model  import Ridge,Lasso,RidgeCV, LassoCV, ElasticNet, ElasticNetCV,LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor 
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score


# In[80]:


df1=df.copy()


# In[81]:


df1.head()


# In[82]:


plt.figure(figsize=(20,25), facecolor='white')
plotnumber = 1

for column in df1:
    if plotnumber<=16 :
        ax = plt.subplot(4,4,plotnumber)
        sns.distplot(df1[column])
        plt.xlabel(column,fontsize=20)
    plotnumber+=1
plt.tight_layout()


# In[83]:


y=df1["Machine failure"]
x=df1.drop(columns=["Machine failure"])


# In[84]:


plt.figure(figsize=(20,25), facecolor='white')
plotnumber = 1

for column in x:
    if plotnumber<=9 :
        ax = plt.subplot(3,3,plotnumber)
        sns.stripplot(y,x[column])
    plotnumber+=1
plt.tight_layout()


# In[85]:


from sklearn.preprocessing import StandardScaler
scaler =StandardScaler()
X_scaled = scaler.fit_transform(x)
X_scaled


# In[86]:


vif = pd.DataFrame()
vif["vif"] = [variance_inflation_factor(X_scaled,i) for i in range(X_scaled.shape[1])]
vif["Features"] = x.columns

#let's check the values
vif


# In[87]:


col=x.columns


# In[88]:


X_scaled=pd.DataFrame(data=X_scaled,columns=col)


# In[89]:


X_scaled


# In[90]:


X_scaled.drop('Torque [Nm]',axis=1,inplace=True)
X_scaled.head()


# In[91]:


X1_train, X1_test, Y1_train, Y1_test = train_test_split(X_scaled, y, test_size=0.3, random_state=0)


# In[92]:


log_reg = LogisticRegression()

log_reg.fit(X1_train,Y1_train)


# In[93]:


y1_predict=log_reg.predict(X1_train)


# In[94]:


accuracy = accuracy_score(Y1_train,y1_predict)
accuracy


# In[95]:


y2_predict=log_reg.predict(X1_test)


# In[96]:


accuracy = accuracy_score(Y1_test,y2_predict)
accuracy


# In[97]:


confusion_matrix=confusion_matrix(Y1_test,y2_predict)


# In[98]:


confusion_matrix


# In[99]:


true_positive = confusion_matrix[0][0]
false_positive = confusion_matrix[0][1]
false_negative = confusion_matrix[1][0]
true_negative = confusion_matrix[1][1]


# In[100]:


# Breaking down the formula for Accuracy
Accuracy = (true_positive + true_negative) / (true_positive +false_positive + false_negative + true_negative)
Accuracy


# In[101]:


# Precison
Precision = true_positive/(true_positive+false_positive)
Precision


# In[102]:


# Recall
Recall = true_positive/(true_positive+false_negative)
Recall


# In[103]:


# F1 Score
F1_Score = 2*(Recall * Precision) / (Recall + Precision)
F1_Score


# In[104]:


# Area Under Curve
auc1 = roc_auc_score(Y1_test, y2_predict)
auc1


# **ROC**

# In[105]:


fpr, tpr, thresholds = roc_curve(Y1_test, y2_predict)


# In[106]:


plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--',label='ROC curve (area = %0.2f)' % auc1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()


# ### Result of Logistic Regression Model :- From the above model, it is conclude that Machine Failure will not happen due to use of Predictive Maintenance.
# 

# In[107]:


import pickle
pickle.dump(log_reg, open('Project3(SL).pkl','wb'))


# In[110]:


maintainance = pickle.load(open('Project3(SL).pkl','rb'))
print(maintainance.predict([[2,295,308,1108,29,1,0,1,1,1]]))


# In[ ]:




