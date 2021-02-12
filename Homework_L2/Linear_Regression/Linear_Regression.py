#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import train_test_split
from sklearn import linear_model
import pandas
from sklearn.metrics import r2_score

data = pandas.read_csv("USA_Housing.csv");
#clean data with missing features rows that have not all features
data = data.dropna(axis=0)

X = data.copy()
y = data['Price'].copy()

print(data.columns)

#remove unrelated data from X, labeled data
toDrop = {'Address','Price'}
X.drop(toDrop, inplace=True, axis = 1)

print(X);
print(y);

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regr = linear_model.LinearRegression()
y_pred = regr.fit(X_train, y_train).predict(X_test)

#Coefficient of determination works like accuracy if it's 1 it's perfect
print('Coefficient of determination %.4f' % r2_score(y_test, y_pred))


# In[ ]:




