import numpy as np
import matplotlib.pyplot as plt
import DataPreprocess as dp
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import SGD
from keras.utils import plot_model

fileName = "features.test.txt"

def initNet(inputShape,outputShape,numUnits,learningRate):
    model = Sequential()
    model.add(Dense(numUnits, input_shape=(inputShape,), activation='sigmoid', kernel_initializer='lecun_uniform'))
    model.add(Dense(outputShape, activation='sigmoid', kernel_initializer='lecun_uniform'))
    model.compile(loss='binary_crossentropy', optimizer=SGD(learning_rate=learningRate), metrics=['accuracy'])
    model.summary()
    picName='P2model' + str(numUnits)+'.png'
    plot_model(model, to_file = picName, show_shapes=True)
    return model

if __name__ == "__main__":
    
    X, Y = dp.GetOneFive(fileName)

    # creat models
    model1 = initNet(2,1,10,0.01)
    model2 = initNet(2,1,6,0.01)
    model3 = initNet(2,1,3,0.01)

    # load models
    model1.set_weights(load_model("P2-1.h5").get_weights())
    model2.set_weights(load_model("P2-2.h5").get_weights())
    model3.set_weights(load_model("P2-3.h5").get_weights())

    # evaluate the model
    loss1, accuracy1 = model1.evaluate(X,Y)
    loss2, accuracy2 = model2.evaluate(X,Y)
    loss3, accuracy3 = model3.evaluate(X,Y)

    print("loss")
    print(loss1,loss2,loss3)
    print("accuracy")
    print(accuracy1,accuracy2,accuracy3)
