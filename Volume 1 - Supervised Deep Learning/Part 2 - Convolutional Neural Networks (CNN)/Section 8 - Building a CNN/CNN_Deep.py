# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 14:23:20 2020

@author: Bibek77
"""

#CNN-------------------------
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense



# Initialising the CNN
classifier = Sequential()
# First convolutional layer
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

#Pooling-------------------
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Flattening--------
classifier.add(Flatten())

#------Full connection-------------
classifier.add(Dense(output_dim=128,activation='relu'))
#---outputlayer is binary so we use sigmoid activation  function we need one node in output layer so dim=1 
classifier.add(Dense(output_dim=1,activation='sigmoid'))

#----compiling CNN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#--------Fitting the CNN To the images------------------------


from keras.preprocessing.image import  ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                    target_size=(64, 64),#size as by our CNN model
                                                    batch_size=32,
                                                    class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')



classifier.fit_generator( training_set,
                        steps_per_epoch=8000,
                        epochs=25,
                        validation_data=test_set,
                         validation_steps=2000)


#-------------making predictions
import numpy as np
from keras.preprocessing import image
NewTest_Image=image.load_img('dataset/newData/cat_or_dog_1.jpg',target_size=(64, 64)) 
NewTest_Image=image.img_to_array(NewTest_Image)#change fro 2d to 3d array 
NewTest_Image=np.expand_dims(NewTest_Image,axis=0)
result=classifier.predict(NewTest_Image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'









