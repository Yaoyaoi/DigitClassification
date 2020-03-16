import numpy as np
import matplotlib.pyplot as plt
import DataPreprocess as dp
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.model_selection import StratifiedKFold

fileName = "zip.train.txt"
fileNameTest = "zip.test.txt"
N = 200


def initNet(inputShape,outputShape,numUnits,learningRate):
    model = Sequential()
    model.add(Dense(numUnits, input_shape=(inputShape,), activation='relu', kernel_initializer='lecun_uniform'))
    model.add(Dense(2, activation='relu', kernel_initializer='lecun_uniform'))
    model.add(Dense(outputShape, activation='sigmoid', kernel_initializer='lecun_uniform'))
    model.compile(loss='binary_crossentropy', optimizer= SGD(learning_rate=learningRate), metrics=['accuracy'])
    return model



if __name__ == "__main__":
# get data with label 1 and 5
    # train data
    X, Y = dp.GetOneFive(fileName)
    # test data
    XTest, YTest = dp.GetOneFive(fileNameTest)

    # create the model
    model = initNet(256,1,6,0.01)



    # fit model1

    model_log = model.fit(X, Y, epochs=N, validation_data=(XTest,YTest))

    Ein = model_log.history['loss']
    Eval = model_log.history['val_loss']

   # plot the data
    logEin = np.log(Ein)
    logEval = np.log(Eval)
    plt.xlabel('iteration')
    plt.ylabel('log10(Error)')
    plt.plot(np.arange(N),logEin,label = "Ein")
    plt.plot(np.arange(N),logEval,label = "Eval")
    plt.legend()
    plt.show()
    




