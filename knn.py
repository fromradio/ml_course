#! /usr/bin/env python
# Author: Ruimin Wang

import math
import sys
import numpy as np
def distance(pt1, pt2):
    dis = math.sqrt(sum([(x-y)*(x-y) for x,y in zip(pt1, pt2)]))
    return dis

# Find the k nearest neighborhood of a given point
#   Simple sort solution
def simpleKNearestNeighbor(pt, k, data, label):
    # very simple solution by sorting the dataset
    sorted_data = sorted(zip(data, label), key = lambda x: distance(x[0], pt))
    # return sorted (x,y)
    return zip(*sorted_data[:k])

# compute the proba of prediction
def computeProba(labels, labeldict):
    result = np.zeros(len(labeldict))
    for label in labels:
        result[labeldict[label]] += 1.0
    result = result * (1.0 / len(labels))
    return result

class SimpleKNN:
    # initialize SimpleKNN algorithm
    #   k: the size of nearest neighborhood
    def __init__(self, k = 10):
        self.k = k
        pass
    def fit(self, x, y):
        self.x = x
        self.y = y
        self.labeldict = {}
        self.reversedict = {}
        i = 0
        for label in set(self.y):
            self.labeldict[label] = i
            self.reversedict[i] = label
            i += 1
    def predict(self, x):
        probas = self.predictProba(x)
        labelids = [np.argmax(p) for p in probas]
        labels = [self.reversedict[l] for l in labelids]
        return labels
    def predictProba(self, x):
        probas = np.zeros((len(x), len(self.labeldict)))
        i = 0
        for data in x:
            pts, labels = simpleKNearestNeighbor(data, self.k, self.x, self.y)
            proba = computeProba(labels, self.labeldict)
            probas[i:] = proba
            i += 1
        return probas

def main():
    dataset = [[0, 0], [1, 0]]
    label = [1,0]
    x,y = simpleKNearestNeighbor([1, 1], 1, dataset, label)
    print x
    print y
    knn = SimpleKNN(1)
    knn.fit(dataset, label)
    print knn.predictProba([[1,1]])
    print knn.predict([[1,1]])

if __name__ == '__main__':
    main()
