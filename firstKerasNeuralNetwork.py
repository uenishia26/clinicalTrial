import tensorflow as tf
import numpy as np 
from random import randint 
from sklearn.utils import shuffle
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, accuracy_score #make_scorer allows you to create custom scoring function that can be used for all scoring parameters
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from keras.layers import LeakyReLU
LeakyReLU = LeakyReLU(alpha=0.1)
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from bayes_opt import BayesianOptimization
from keras.wrappers.scikit_learn import KerasClassifier
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
scaled_train_sample = scaler.fit_transform(train_samples.reshape(-1,1))
scaled_test_sample = scaler.fit_transform(test_samples.reshape(-1,1)) 

#Hyper Tunning Values

def baysOptimizer(neurons, activation, optimizer, learning_rate, batch_size, epoch):
    activationF = ['relu', 'sigmoid', 'softplus', 'softsign', 'tanh', 'selu',
                    'elu', 'exponential', LeakyReLU,'relu']

    optimizerD = {"Adam":Adam(learning_rate), "SGD":SGD(learning_rate), "RMSprop":RMSprop(learning_rate),
                "Adadelta":Adadelta(learning_rate), "Adagrad":Adagrad(learning_rate), 
                "Adamax":Adamax(learning_rate), "Nadam":Nadam(learning_rate), "Ftrl":Ftrl(learning_rate)}
    optimizerL = ['SGD', 'Adam', 'RMSprop', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl']

    #Bayeisan might output non integer optimal values
    activation = activationF[round(activation)] #Round and find the equivalency
    neurons = round(neurons)
    batch_size = round(batch_size)
    epochs = round(epochs)
    optimizerL = round(optimizer)

    #Creates the hypertunned mode. 
    def createModel():
        model = Sequential()
        model.add(Dense(units=neurons, input_dim=(1,), activation=activation))
        model.add(Dense(units=neurons, activation=activation))
        model.add(Dense(units=2, activation='softmax'))
        model.compile(optimizer=optimizerD[optimizerL],loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model 

    model = KerasClassifier(build_fn=createModel, epochs=epoch, batch_size=batch_size) #Wrapping the model so sklearn use
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    #This is for testing 
    score = cross_val_score(model,scaled_train_sample, train_labels, cv=skf, scoring=custom_accuracy)
    return score 



params_nn = {
    'neurons': (10, 100),
    'activation':(0, 9),
    'optimizer':(0,7),
    'learning_rate':(0.01, 1),
    'batch_size':(200, 1000),
    'epochs':(20, 100)
}

baysOpt = BayesianOptimization(baysOptimizer, params_nn, random_state=42)
baysOpt.maximize(init_points=25, n_iter=4) #Use 25 random select iterations to create surrogate model, 4 optimizing iterations after


#rounded_predictions = np.argmax(predictions, axis=-1)



