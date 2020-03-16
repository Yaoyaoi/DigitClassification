import numpy as np
import matplotlib.pyplot as plt
import DataPreprocess as dp
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.optimizers import Adadelta
from sklearn.model_selection import StratifiedKFold
from keras.utils import to_categorical
from keras.utils import plot_model

# filename of data  
fileName = "zip.test.txt"

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
    # plot the picture of the model
    plot_model(model, to_file = 'P4model.png', show_shapes=True)

    model.compile(loss='categorical_crossentropy', optimizer= Adadelta(learning_rate=learningRate), metrics=['accuracy'])
    return model


if __name__ == "__main__":

# data preprocessing
    data = dp.LoadData(fileName)
    X = np.array(data)[:,1:]
    Xtest = X.reshape((np.size(X,0),16,16,1))
    Y = np.array(data)[:,0]
    Ytest = to_categorical(Y,10)

# create model
    model = initNet(10,1)
# load model
    model.set_weights(load_model("P4.h5").get_weights())
# evaluate
    loss, acc = model.evaluate(Xtest,Ytest)
    print(acc)

