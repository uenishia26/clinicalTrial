import tensorflow as tf
import numpy as np 
from random import randint 
from sklearn.utils import shuffle
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, accuracy_score #make_scorer allows you to create custom scoring function that can be used for all scoring parameters
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Input
from keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from keras.layers import LeakyReLU
import keras_tuner as kt
from tensorflow.keras.utils import to_categorical
LeakyReLU = LeakyReLU(negative_slope=0.1)
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from bayes_opt import BayesianOptimization, UtilityFunction
from scikeras.wrappers import KerasClassifier
import matplotlib.pyplot as plt 

#Customized accuracy 
def custom_accuracy(ytrue, ypred):
    rounded_predictions = np.argmax(ypred, axis=-1) #Rounding the predictions to either 1 or 0
    return accuracy_score(ytrue, ypred)
custom_scorer = make_scorer(custom_accuracy, greater_is_better=True) #Higher accuracy indidcates better performance 
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

#XValues (sample)
scaled_train_sample = scaler.fit_transform(train_samples.reshape(-1,1))
scaled_test_sample = scaler.fit_transform(test_samples.reshape(-1,1)) 

#YValues (labels) HotEncodingTheseLabels
train_labels = tf.keras.utils.to_categorical(train_labels,num_classes=2)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=2)


#Hyper Tunning Values

def baysOptimizer(hp):
    #neurons, activation, optimizer, learning_rate, batch_size, epoch

    #Bayeisan might output non integer optimal values
    numNeuros = hp.Int('units', min_value=10, max_value=100, step=1)
    activation = hp.Choice('activation', ['relu', 'sigmoid', 'softplus', 'softsign', 'tanh', 'selu',
                    'elu', 'exponential','relu']) #LeakReLu Implementation
    optimizer_choice = hp.Choice('optimizer', ['SGD', 'Adam', 'RMSprop', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl'])
    learning_rate = hp.Float('lr', min_value=0.0001, max_value=0.1, sampling='log') #Better for fine tunning at lower level

    # Dynamic optimizer selection
    if optimizer_choice == 'SGD':
        optimizer = SGD(learning_rate=learning_rate)
    elif optimizer_choice == 'Adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_choice == 'RMSprop':
        optimizer = RMSprop(learning_rate=learning_rate)
    elif optimizer_choice == 'Adadelta':
        optimizer = Adadelta(learning_rate=learning_rate)
    elif optimizer_choice == 'Adagrad':
        optimizer = Adagrad(learning_rate=learning_rate)
    elif optimizer_choice == 'Adamax':
        optimizer = Adamax(learning_rate=learning_rate)
    elif optimizer_choice == 'Nadam':
        optimizer = Nadam(learning_rate=learning_rate)
    elif optimizer_choice == 'Ftrl':
        optimizer = Ftrl(learning_rate=learning_rate)

    model = Sequential()
    model.add(Input(shape=(1,)))
    model.add(Dense(units=numNeuros, activation=activation))
    model.add(Dense(units=numNeuros, activation=activation))
    model.add(Dense(units=2, activation='softmax')) #Final output 

    model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])

    return model 

tuner = kt.BayesianOptimization(hypermodel=baysOptimizer,
                                objective='accuracy', 
                                max_trials = 45,
                                num_initial_points=25, 
                                directory='dir',
                                project_name='x') #Trains Gaussian process model
    

tuner.search(scaled_train_sample, train_labels, validation_split=0.2, epochs=10) #Same as fit function


