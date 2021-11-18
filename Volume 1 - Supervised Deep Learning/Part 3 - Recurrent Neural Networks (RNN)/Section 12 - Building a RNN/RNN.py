# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 19:33:47 2020

@author: Bibek77
"""

#------------RNN using LSTM--------------------------------
#-------------importing the basis libraries----------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#----------------------------------------------


#Part I ----Importing the training set--------------------------------------
dataset_train=pd.read_csv('Google_Stock_Price_Train.csv',delimiter=';')
training_set=dataset_train.iloc[:,1:2].values


# Feature scaling using normalisation 
from sklearn.preprocessing import  MinMaxScaler
F_scaling=MinMaxScaler(feature_range=(0,1))
training_set_scaled=F_scaling.fit_transform(training_set)

#creating a data structure with 60 timesteps and 1 output
X_train=[]
Y_train=[]

for i in range(60,1258):
    X_train.append(training_set_scaled[i-60:i,0])
    Y_train.append(training_set_scaled[i,0])

X_train,Y_train=np.array(X_train),np.array(Y_train)
    

#---reshaping-----

X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))



#Part I ----Importing the training set--------------------------------------




#Part II ----Building RNN---------------------------------------

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#Initialising the RNN
RNN_regressor=Sequential()

#Adding first LSTM Layer

RNN_regressor.add(LSTM(units=50,return_sequences=True, input_shape=(X_train.shape[1],1)))
RNN_regressor.add(Dropout(0.2))

#Adding second LSTM Layer 
RNN_regressor.add(LSTM(units=50,return_sequences=True))
RNN_regressor.add(Dropout(0.2))

#Adding third LSTM Layer 
RNN_regressor.add(LSTM(units=50,return_sequences=True))
RNN_regressor.add(Dropout(0.2))

#Adding fourth  LSTM Layer 
RNN_regressor.add(LSTM(units=50))
RNN_regressor.add(Dropout(0.2))


#---Adding output layer----
RNN_regressor.add(Dense(units=1))      #dense function is used to  make full connection



#Part II ----Building RNN--------------------------------------

###-----Compiling the RNN-----
RNN_regressor.compile(optimizer='adam',loss='mean_squared_error')


#Fitting th RNN to the Training  set 

RNN_regressor.fit(X_train ,Y_train ,epochs=100,batch_size=32)

#Part III ----Making the prediction and visualising the results---------------------------

#getting the real stock price  


dataset_test=pd.read_csv('Google_Stock_Price_Test.csv',delimiter=';')
real_stock_price=dataset_test.iloc[:,1:2].values

#predict real stock price 
dataset_total=pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)

inputs=dataset_total[len(dataset_total)-len(dataset_test)-60:].values
inputs=inputs.reshape(-1,1)
inputs=F_scaling.transform(inputs)
X_test=[]

for i in range(60,80):
    X_test.append(inputs[i-60:i,0])
  

X_test=np.array(X_test)
X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
pred_stock_price =RNN_regressor.predict(X_test)
pred_stock_price =F_scaling.inverse_transform(pred_stock_price) 

#Part III ----Making the prediction and visualising the results---------------------------



# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(pred_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()























