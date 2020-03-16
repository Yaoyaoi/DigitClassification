import numpy as np
import matplotlib.pyplot as plt
import DataPreprocess as dp
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.model_selection import StratifiedKFold

fileName = "zip.train.txt"
N = 100


def initNet(inputShape,outputShape,numUnits,learningRate):
    model = Sequential()
    model.add(Dense(numUnits, input_shape=(inputShape,), activation='relu', kernel_initializer='lecun_uniform'))
    model.add(Dense(2, activation='relu', kernel_initializer='lecun_uniform'))
    model.add(Dense(outputShape, activation='sigmoid', kernel_initializer='lecun_uniform'))
    model.compile(loss='binary_crossentropy', optimizer= SGD(learning_rate=learningRate), metrics=['accuracy'])
    #plot_model(model, to_file='model.png',show_shapes=True)
    return model



if __name__ == "__main__":
# get data with label 1 and 5
    X, Y = dp.GetOneFive(fileName)

# 3-fold cross-validation
    inSampleErr = []
    testSetErr = []
    inSampleAcc = []
    testSetAcc = []
    kfold = StratifiedKFold(n_splits=3, shuffle=True)
    for train, test in kfold.split(X, Y):
        # init the model
        model1 = initNet(256,1,6,0.01)
        model2 = initNet(256,1,3,0.01)

        # fit the model
        model1.fit(X[train], Y[train], epochs=N)
        model2.fit(X[train], Y[train], epochs=N)

        # evaluate the model

        loss, accuracy = model1.evaluate(X[train], Y[train])
        inSampleErr.append(loss)
        inSampleAcc.append(accuracy)
        loss, accuracy = model1.evaluate(X[test], Y[test])
        testSetErr.append(loss)
        testSetAcc.append(accuracy)

        loss, accuracy = model2.evaluate(X[train], Y[train])
        inSampleErr.append(loss)
        inSampleAcc.append(accuracy)
        loss, accuracy = model2.evaluate(X[test], Y[test])
        testSetErr.append(loss)
        testSetAcc.append(accuracy)

# train two models with all the train data
    # create models
    model1 = initNet(256,1,6,0.01)
    model2 = initNet(256,1,3,0.01)

    # fit models
    model1.fit(X, Y, epochs=N)
    model2.fit(X, Y, epochs=N)

    # save the model
    model1.save('P3-1.h5')
    model2.save('P3-2.h5')

# print Err
    print("inSampleErr")
    print(inSampleErr)
    print("inSampleAcc")
    print(inSampleAcc)
    print("testSetErr")
    print(testSetErr)
    print("testSetAcc")
    print(testSetAcc)


# calculate the mean error of each model 
    inSampleErrMean = []
    inSampleAccMean = []
    testSetErrMean = []
    testSetAccMean = []


    for i in range(2):
        inErrMean = (inSampleErr[i]+inSampleErr[i+2]+inSampleErr[i+4])/3
        inSampleErrMean.append(inErrMean)
        inAccMean = (inSampleAcc[i]+inSampleAcc[i+2]+inSampleAcc[i+4])/3
        inSampleAccMean.append(inAccMean)
        teErrMean = (testSetErr[i]+testSetErr[i+2]+testSetErr[i+4])/3
        testSetErrMean.append(teErrMean)
        testAccMean = (inSampleAcc[i]+inSampleAcc[i+2]+inSampleAcc[i+4])/3
        testSetAccMean.append(testAccMean)
    
    print("inSampleErrMean")
    print(inSampleErrMean)
    print("inSampleAccMean")
    print(inSampleAccMean)
    print("testSetErrMean")
    print(testSetErrMean)
    print("testSetAccMean")
    print(testSetAccMean)