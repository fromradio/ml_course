#! /usr/bin/env python
# Author: Ruimin Wang

# useful tools for machine learning toolbox


import numpy as np
import sys

# Read from a given dataset
#   DataFormat: each feature is joined with a specified separator
#   and the last one is label
#   The feature are transformed into numbers automatically
def readDataSet(filename, separator = ','):
    with open(filename) as fp:
        result = [[]] 
        labels = np.array([])
        first = True
        for line in fp:
            parts = line.rstrip().split(separator)
            if first:
                result = [[float(x) for x in parts[:len(parts)-1]]]
                first = False
            else:
                result = np.append(result, [[float(x) for x in parts[:len(parts)-1]]], axis = 0)
            labels = np.append(labels, [parts[-1]])
        return result, labels

# Split the data into train and test vector
def splitDataToTrainTest(x,y, prob = 0.7):
    trainx = []
    trainy = []
    testx = []
    testy = []
    firstTrain = True
    firstTest = True
    for i in range(0, len(y)):
        if np.random.rand() < prob:
            trainx.append(x[i])
            trainy.append(y[i])
        else:
            testx.append(x[i])
            testy.append(y[i])
    return np.array(trainx), np.array(trainy), np.array(testx), np.array(testy)

if __name__ == '__main__':
    data, label = readDataSet('test.txt')
    print data
    print label
    trainx, trainy, testx, testy = splitDataToTrainTest(data, label)
    print trainx, trainy
