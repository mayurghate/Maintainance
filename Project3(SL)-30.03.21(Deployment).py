import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

os.chdir(r'F:\Mayur Ghate\Python Programing\Python by Uday Sir\Assignment\POC Project')

df = pd.read_csv('ai4i2020.csv')

df.head()

df.shape

df.info()

df.isna().sum()

df.drop(['UDI','Product ID'],axis=1,inplace=True)

df['Type'].unique()

def type(x):
    if x=='L':
        return 1
    elif x=='M':
        return 2
    else:
        return 3
    
df['Type']=df['Type'].map(type)

df.head()

df.describe()

sns.distplot(df)

df.columns

df.boxplot(figsize=(18,10))

df['Rotational speed [rpm]'].plot(kind='box')

df[['TWF', 'HDF', 'PWF', 'OSF', 'RNF']].sum().sort_values(ascending=False).plot(kind='bar')
df[['TWF', 'HDF', 'PWF', 'OSF', 'RNF']].sum().sort_values(ascending=False)

df[['TWF', 'HDF', 'PWF', 'OSF', 'RNF']].sum().sum()

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model  import Ridge,Lasso,RidgeCV, LassoCV, ElasticNet, ElasticNetCV,LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor 
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score

df1=df.copy()

df1.head()

plt.figure(figsize=(20,25), facecolor='white')
plotnumber = 1

for column in df1:
    if plotnumber<=16 :
        ax = plt.subplot(4,4,plotnumber)
        sns.distplot(df1[column])
        plt.xlabel(column,fontsize=20)
    plotnumber+=1
plt.tight_layout()

y=df1["Machine failure"]
x=df1.drop(columns=["Machine failure"])

plt.figure(figsize=(20,25), facecolor='white')
plotnumber = 1

for column in x:
    if plotnumber<=9 :
        ax = plt.subplot(3,3,plotnumber)
        sns.stripplot(y,x[column])
    plotnumber+=1
plt.tight_layout()

from sklearn.preprocessing import StandardScaler
scaler =StandardScaler()
X_scaled = scaler.fit_transform(x)
X_scaled

vif = pd.DataFrame()
vif["vif"] = [variance_inflation_factor(X_scaled,i) for i in range(X_scaled.shape[1])]
vif["Features"] = x.columns
vif

col=x.columns

X_scaled=pd.DataFrame(data=X_scaled,columns=col)

X_scaled

X_scaled.drop('Torque [Nm]',axis=1,inplace=True)
X_scaled.head()

X1_train, X1_test, Y1_train, Y1_test = train_test_split(X_scaled, y, test_size=0.3, random_state=0)

log_reg = LogisticRegression()
log_reg.fit(X1_train,Y1_train)

y1_predict=log_reg.predict(X1_train)

accuracy = accuracy_score(Y1_train,y1_predict)
accuracy

y2_predict=log_reg.predict(X1_test)

accuracy = accuracy_score(Y1_test,y2_predict)
accuracy

confusion_matrix=confusion_matrix(Y1_test,y2_predict)

confusion_matrix

true_positive = confusion_matrix[0][0]
false_positive = confusion_matrix[0][1]
false_negative = confusion_matrix[1][0]
true_negative = confusion_matrix[1][1]

Accuracy = (true_positive + true_negative) / (true_positive +false_positive + false_negative + true_negative)
Accuracy

Precision = true_positive/(true_positive+false_positive)
Precision

Recall = true_positive/(true_positive+false_negative)
Recall

F1_Score = 2*(Recall * Precision) / (Recall + Precision)
F1_Score

auc1 = roc_auc_score(Y1_test, y2_predict)
auc1

fpr, tpr, thresholds = roc_curve(Y1_test, y2_predict)

plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--',label='ROC curve (area = %0.2f)' % auc1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()