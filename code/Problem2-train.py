import numpy as np
import matplotlib.pyplot as plt
import DataPreprocess as dp
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.model_selection import StratifiedKFold


fileName = "features.train.txt"
N = 100

def initNet(inputShape,outputShape,numUnits,learningRate):
    model = Sequential()
    model.add(Dense(numUnits, input_shape=(inputShape,), activation='sigmoid', kernel_initializer='lecun_uniform'))
    model.add(Dense(outputShape, activation='sigmoid', kernel_initializer='lecun_uniform'))
    model.compile(loss='binary_crossentropy', optimizer=SGD(learning_rate=learningRate), metrics=['accuracy'])
    return model


if __name__ == "__main__":
# data preprocess
    X, Y = dp.GetOneFive(fileName)
    inSampleErr = []
    testSetErr = []
    inSampleAcc = []
    testSetAcc = []

# 3-fold cross-validation
    kfold = StratifiedKFold(n_splits=3, shuffle=True)
    for train, test in kfold.split(X, Y):
        # creat model
        model1 = initNet(2,1,10,0.01)
        model2 = initNet(2,1,6,0.01)
        model3 = initNet(2,1,2,0.01)

        # fit the model
        model1.fit(X[train], Y[train], epochs=N)
        model2.fit(X[train], Y[train], epochs=N)
        model3.fit(X[train], Y[train], epochs=N)

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

        loss, accuracy = model3.evaluate(X[train], Y[train])
        inSampleErr.append(loss)
        inSampleAcc.append(accuracy)
        loss, accuracy = model3.evaluate(X[test], Y[test])
        testSetErr.append(loss)
        testSetAcc.append(accuracy)


# train three models with all the train data
    model1 = initNet(2,1,10,0.25)
    model2 = initNet(2,1,6,0.25)
    model3 = initNet(2,1,3,0.25)

    # fit the model
    model1.fit(X, Y, epochs=N)
    model2.fit(X, Y, epochs=N)
    model3.fit(X, Y, epochs=N)

    # save the model
    model1.save("P2-1.h5")
    model2.save("P2-2.h5")
    model3.save("P2-3.h5")

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

    for i in range(3):
        inErrMean = (inSampleErr[i]+inSampleErr[i+3]+inSampleErr[i+6])/3
        inSampleErrMean.append(inErrMean)
        inAccMean = (inSampleAcc[i]+inSampleAcc[i+3]+inSampleAcc[i+6])/3
        inSampleAccMean.append(inAccMean)
        teErrMean = (testSetErr[i]+testSetErr[i+3]+testSetErr[i+6])/3
        testSetErrMean.append(teErrMean)
        testAccMean = (inSampleAcc[i]+inSampleAcc[i+3]+inSampleAcc[i+6])/3
        testSetAccMean.append(testAccMean)
    
    print("inSampleErrMean")
    print(inSampleErrMean)
    print("inSampleAccMean")
    print(inSampleAccMean)
    print("testSetErrMean")
    print(testSetErrMean)
    print("testSetAccMean")
    print(testSetAccMean)
    
    
    