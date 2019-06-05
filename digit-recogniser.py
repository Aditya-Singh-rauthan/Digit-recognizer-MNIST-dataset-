# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 01:21:10 2019

@author: Aditya Rauthan
"""

import pandas as pd
import numpy as np

train=pd.read_csv('train1.csv')
test=pd.read_csv('test1.csv')

imageId=test.index+1

print(imageId)
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier as kn
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

train = train.sample(frac=1)
train = train.head(5000)

x=train.drop('label',axis=1)
y=train['label']

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=101)

st_scaler=StandardScaler()

x_train=st_scaler.fit_transform(x_train)
x_test=st_scaler.transform(x_test)

model=kn()
model.fit(x_train,y_train)
predictions=model.predict(x_test)

print(accuracy_score(y_test,predictions))

x=model.predict(test)
l={'ImageId':imageId,'label':x}
df=pd.DataFrame(l)
df.to_csv('sub_for_dig.csv',index=False)