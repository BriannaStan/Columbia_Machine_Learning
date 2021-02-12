#!/usr/bin/env python
# coding: utf-8

# In[2]:


import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas

data = pandas.read_csv("dataset-tortuga.csv");
#clean data with missing features
data = data.dropna(axis=0)

X = data.copy()
y = data['PROFILE'].copy()

#remove unrelated data from X, labeled data
toDrop = {'ID','NAME','USER_ID','PROFILE'}
X.drop(toDrop, inplace=True, axis = 1)

print(X);
print(y);


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

clas = RandomForestClassifier()
startt = time.time()
y_pred = clas.fit(X_train, y_train).predict(X_test)
misses = (y_test != y_pred).sum()
total = X_test.shape[0]
endt = time.time()
print("Accuracy:%f%% in %f seconds" %( (total-misses)/total*100.0, endt-startt))


# In[ ]:




