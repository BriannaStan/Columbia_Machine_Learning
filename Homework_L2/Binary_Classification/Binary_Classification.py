#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

unfilteredData = pandas.read_csv("weatherAUS.csv")
print(unfilteredData.columns)
data = unfilteredData[(unfilteredData.RainTomorrow == "Yes" )|(unfilteredData.RainTomorrow == "No")]
data = data.dropna(axis=0)

X = data.copy()
y = data['RainTomorrow'].map({'Yes':1,'No':0}).copy()

#remove unrelated data from X, labeled info and Date
toDrop = {'RainTomorrow','RainToday','Date'}
X.drop(toDrop, inplace=True, axis = 1)

#Locations are strings, make them numbers
le = preprocessing.LabelEncoder()
locations = X['Location'].unique()
le.fit(locations)
X['Location']=le.transform(X['Location'])
locations = X['Location'].unique()

#WindGustDirs are strings, make them numbers
WindGustDirs = X['WindGustDir'].unique()
le.fit(WindGustDirs)
X['WindGustDir']=le.transform(X['WindGustDir'])

#WindGustDirs are strings, make them numbers
WindDir9ams = X['WindDir9am'].unique()
le.fit(WindDir9ams)
X['WindDir9am']=le.transform(X['WindDir9am'])

#WindGustDirs are strings, make them numbers
WindDir3pms = X['WindDir3pm'].unique()
le.fit(WindDir3pms)
X['WindDir3pm']=le.transform(X['WindDir3pm'])

print(X);
print(y);

#we have a big set of data so we take 25% of it for testing and the rest of 75% will be used for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

clas = {}
clas['KNeighbours'] = KNeighborsClassifier()
clas['DecisionTree'] = DecisionTreeClassifier()
clas['GaussianNB'] = GaussianNB()
for k in clas.keys():
    startt = time.time()
    c = clas.get(k)
    y_pred = c.fit(X_train, y_train).predict(X_test)
    print(y_pred[:100])
    misses = (y_test != y_pred).sum()
    total = X_test.shape[0]
    endt = time.time()
    print("Accuracy for %s :%f%% in %f seconds" %( k, (total-misses)/total*100.0, endt-startt))


# In[ ]:




