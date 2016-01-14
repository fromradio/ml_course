#! /usr/bin/env python
# Author: Ruimin Wang

import utils
import knn

# Train data
if __name__ == '__main__':
    x, y = utils.readDataSet('test.txt')
    trainx, trainy, testx, testy = utils.splitDataToTrainTest(x,y)
    model = knn.SimpleKNN(7)
    model.fit(trainx, trainy)
    predict = model.predict(testx)
    
    print predict
    print sum([1 for i in range(0,len(testy)) if predict[i] == testy[i]])
    print len(testy)
