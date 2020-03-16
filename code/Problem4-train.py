import numpy as np
import matplotlib.pyplot as plt
import DataPreprocess as dp
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.optimizers import Adadelta
from sklearn.model_selection import StratifiedKFold
from keras.utils import to_categorical

# filename of data  
fileName = "zip.train.txt"

# The number of iterations
N = 10

# creat the model
def initNet(numCategory,learningRate):

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(16,16,1)))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(numCategory, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer= Adadelta(learning_rate=learningRate), metrics=['accuracy'])
    return model


if __name__ == "__main__":

# data preprocessing
    data = dp.LoadData(fileName)
    X = np.array(data)[:,1:]
    X = X.reshape((np.size(X,0),16,16,1))
    Y = np.array(data)[:,0]
#    img = X[0]
#    plt.imshow(img, cmap='Greys', interpolation='nearest')
#    plt.show()

# 3-fold cross-validation

    inSampleAcc = []
    testSetAcc = []
    kfold = StratifiedKFold(n_splits=3, shuffle=True)
    for train, test in kfold.split(X, Y):
        Ytrain = to_categorical(Y[train], 10)
        Ytest = to_categorical(Y[test], 10)
        # init the model
        model = initNet(10,1)

        # fit the model
        model.fit(X[train], Ytrain, epochs=N)

        # evaluate the model
        loss, accuracy = model.evaluate(X[train], Ytrain)
        inSampleAcc.append(accuracy)
        loss, accuracy = model.evaluate(X[test], Ytest)
        testSetAcc.append(accuracy)
    
# train on all data
    # data process
    YCate = to_categorical(Y, 10)
    # create model
    model = initNet(10,1)
    # fit the model
    model.fit(X, YCate, epochs=N)
    # save the model
    model.save('P4.h5')

# calculate and print Acc
    print("inSampleAcc")
    print(inSampleAcc)
    print("testSetAcc")
    print(testSetAcc)

    inSampleAccVar = np.var(inSampleAcc)
    testSetAccVar = np.var(testSetAcc)

    print("inSampleAccVar")
    print(inSampleAccVar)
    print("testSetAccVar")
    print(testSetAccVar)