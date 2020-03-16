import numpy as np
import matplotlib.pyplot as plt
import DataPreprocess as dp

fileName = "features.train.txt"

# plot the Points
def PlotPoints(targetNum,dataDivided): 
    plt.figure(targetNum)               # different figures
    colors = ['crimson', 'dodgerblue',  'brown', 'orange', 'chocolate', 'olive', 'gold', 'hotpink', 'indigo', 'darkcyan']
    labels = ['0','1','2','3','4','5','6','7','8','9']
    plt.xlabel('intensity')
    plt.ylabel('symmetry')
    hand = []
    for i in range(10):
        data = np.array(dataDivided[i])
        shape = data.shape
        hand.append(plt.scatter(data[:,1], data[:,2], c = (colors[i] if targetNum==10 else colors[i==targetNum]), s=5))
    plt.legend(hand,labels)


if __name__ == "__main__":
    data = dp.LoadData(fileName)
    dataDivided = dp.DivideData(data)
    
    # plot 11 different figures
    for i in range(11):
        PlotPoints(i,dataDivided)
    plt.show()