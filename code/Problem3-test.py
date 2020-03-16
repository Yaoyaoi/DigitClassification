import numpy as np
import matplotlib.pyplot as plt
import DataPreprocess as dp
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import SGD
from keras.utils import plot_model

fileName = "zip.test.txt"

N = 100


def initNet(inputShape,outputShape,numUnits,learningRate):
    model = Sequential()
    model.add(Dense(numUnits, input_shape=(inputShape,), activation='relu', kernel_initializer='lecun_uniform'))
    model.add(Dense(2, activation='relu', kernel_initializer='lecun_uniform'))
    model.add(Dense(outputShape, activation='sigmoid', kernel_initializer='lecun_uniform'))
    model.compile(loss='binary_crossentropy', optimizer= SGD(learning_rate=learningRate), metrics=['accuracy'])
    picName='P3model' + str(numUnits)+'.png'
    plot_model(model, to_file = picName, show_shapes=True)
    return model



if __name__ == "__main__":
# get data with label 1 and 5
    X, Y = dp.GetOneFive(fileName)
# predict
    # init model
    model1 = initNet(256,1,6,0.01)
    model2 = initNet(256,1,3,0.01)
    # load models
    model1.set_weights(load_model("P3-1.h5").get_weights())
    model2.set_weights(load_model("P3-2.h5").get_weights())
# evaluate
    loss1, accuracy1 = model1.evaluate(X,Y)
    loss2, accuracy2 = model2.evaluate(X,Y)

    print("loss")
    print(loss1,loss2)
    print("accuracy")
    print(accuracy1,accuracy2)

