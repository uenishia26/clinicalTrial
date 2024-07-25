import tensorflow as tf
import numpy as np 
from random import randint 
from sklearn.utils import shuffle
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from sklearn.metrics import classification_report

"""
Data: 
A clinical trial was done for a new drug between the ages of 13 to 100 
    - The trial had 2500 participants. Half under 65. Half over 65
    - 95% of individuals Under 65 didn't experince any side effects
    - 95% of indviduals Over 65 did experince side effects
"""
trainSamples = [] #Indicatesd the age 
trainLabels = [] #1: Did expericne side effects, 0: did not experience side effects 
testSamples = []
testLabels = []

#Train dataset 
#Data for the 5% 
for i in range(50): 
    #5% of Pateints under 65 that did experince side effects 
    random_younger = randint(13,64)
    trainSamples.append(random_younger)
    trainLabels.append(1)

    #5% of 65 and older that didn't experince side effects
    random_older = randint(65, 100)
    trainSamples.append(random_older)
    trainLabels.append(0)

#Data for the 95%
for i in range(1000): 

    #95% of 65 and younger that didn't experince side effects 
    random_younger = randint(13,64)
    trainSamples.append(random_younger)
    trainLabels.append(0)

    #95% of 65 and older that did experince side effects
    random_younger = randint(65,100)
    trainSamples.append(random_younger)
    trainLabels.append(1)

#Test Set 
for i in range(25): 
    #5% of Pateints under 65 that did experince side effects 
    random_younger = randint(13,64)
    testSamples.append(random_younger)
    testLabels.append(1)

    #5% of 65 and older that didn't experince side effects
    random_older = randint(65, 100)
    testSamples.append(random_older)
    testLabels.append(0)

#Data for the 95%
for i in range(250): 

    #95% of 65 and younger that didn't experince side effects 
    random_younger = randint(13,64)
    testSamples.append(random_younger)
    testLabels.append(0)

    #95% of 65 and older that did experince side effects
    random_younger = randint(65,100)
    testSamples.append(random_younger)
    testLabels.append(1)


train_labels = np.array(trainLabels)
train_samples = np.array(trainSamples)
test_labels = np.array(testLabels)
test_samples = np.array(testSamples)
train_labels, train_samples = shuffle(train_labels, train_samples)
test_labels, test_samples = shuffle(test_labels, test_samples)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_train_sample = scaler.fit_transform(train_samples.reshape(-1,1))
scaled_test_sample = scaler.fit_transform(test_samples.reshape(-1,1))



model = Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='softmax')])

#Prepares for trainning
model.compile(optimizer=Adam(learning_rate=0.0001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#trainning
#Note: validation_set is take from the last 10% of the data thus you must shuffle data before calling the fit function 
model.fit(x=scaled_train_sample, y=train_labels, validation_split=0.1, batch_size=10, epochs=30, shuffle=True, verbose=2)

predictions = model.predict(x=scaled_test_sample, batch_size=10)

rounded_predictions = np.argmax(predictions, axis=-1)

classification_report(rounded_predictions, test_labels)
print(classification_report(rounded_predictions, test_labels))