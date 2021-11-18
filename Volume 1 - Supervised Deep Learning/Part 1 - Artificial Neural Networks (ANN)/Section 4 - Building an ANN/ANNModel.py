# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 18:11:46 2020

@author: Bibek77
"""


#----------------------ANN---------------------------------

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv',delimiter=';')


X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values


#----------encoding categorical data(optional)-------
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])


ct = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)
X=X[:,1:]#remove dummy variable


#------------------------------------------------------------------------------------------



#splitting dataset into test set and train set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
SC_X=StandardScaler()
X_train= SC_X.fit_transform(X_train)
X_test= SC_X.fit_transform(X_test)


#-----------Make ANN--------------------------


import keras 
from keras.models import Sequential
from keras.layers import Dense

#initialize ANN
classifier= Sequential()


#Addinput layer and first hiddenlayer  using activation fun rectifier  for hidden layer 
classifier.add(Dense(output_dim=6, init='uniform',activation='relu',input_dim=11))

#Add second  hiddenlayer  using activation fun relu for hidden layer

classifier.add(Dense(output_dim=6, init='uniform',activation='relu'))

#Add output layer using sigmoid function as activation function 

classifier.add(Dense(output_dim=1, init='uniform',activation='sigmoid'))

#compiling ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics= ['accuracy'])

#fitting ANN in traing set---
classifier.fit(X_train,Y_train,batch_size=10, nb_epoch=100)


#predicting the test set result 

Y_pred=classifier.predict(X_test)
Y_pred=(Y_pred >0.5)


#predicting a single new observation 
""" 
Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $60000
Number of Products: 2
Does this customer have a credit card ? Yes
Is this customer an Active Member: Yes
Estimated Salary: $50000
"""



new_pred=classifier.predict(SC_X.transform(np.array([[0.0 , 0 , 600 , 1 , 40 , 3 ,60000 , 2 , 1 , 1 , 50000]])))
new_pred=(new_pred >0.5)



#making confusion matrices
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,Y_pred)









