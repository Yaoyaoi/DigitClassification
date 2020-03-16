import numpy as np

# load the data from file
def LoadData(fileName):     
    fid = open(fileName)
    lines = fid.readlines()
    dataLoad = []
    for line in lines:
        lineSplit = line.strip().split()
        dataLoad.append(list(map(float,lineSplit)))
    return dataLoad

# divide the data into 10 lists by label
def DivideData(data):           
    data0 = []
    data1 = []
    data2 = []
    data3 = []
    data4 = []
    data5 = []
    data6 = []
    data7 = []
    data8 = []
    data9 = []

    for dataPoints in data:
        if dataPoints[0]==0.0:
            data0.append(dataPoints)
        elif dataPoints[0]==1.0:
            data1.append(dataPoints)
        elif dataPoints[0]==2.0:
            data2.append(dataPoints)
        elif dataPoints[0]==3.0:
            data3.append(dataPoints)
        elif dataPoints[0]==4.0:
            data4.append(dataPoints)
        elif dataPoints[0]==5.0:
            data5.append(dataPoints)
        elif dataPoints[0]==6.0:
            data6.append(dataPoints)
        elif dataPoints[0]==7.0:
            data7.append(dataPoints)
        elif dataPoints[0]==8.0:
            data8.append(dataPoints)
        elif dataPoints[0]==9.0:
            data9.append(dataPoints)

    return [data0,data1,data2,data3,data4,data5,data6,data7,data8,data9]

# divide data with label 1 and 5 from all data
def GetOneFive(fileName):
    data = LoadData(fileName)

# get the data with label 1 and 5
    dataOneFive = []
    for dataPoints in data:
        if dataPoints[0]==1.0:
            dataOneFive.append(dataPoints)
        elif dataPoints[0]==5.0:
            dataPoints[0]=0.0
            dataOneFive.append(dataPoints)

# X is the data with label 1 and 5
# Y is the label
    X = np.array(dataOneFive)[:,1:]
    Y = np.array(dataOneFive)[:,0]

    return X,Y
