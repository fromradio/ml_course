#! /usr/bin/env python
# Author: Ruimin Wang

# Decision Tree algorithm
#   ID 3 Tree is implemented
#   Basic assumption:
#       the last dimension of a sample is its label

import math

# compute entropy
#   - sum p_ilog(p_i)
def computeEntropy(vec):
    labelVec = {}
    for featurevec in vec:
        label = featurevec[-1]
        labelVec.setdefault(label, 0)
        labelVec[label] += 1
    N = len(vec)
    result = 0.0
    for l in labelVec:
        v = float(labelVec[l]) / N
        result -= math.log(v, 2.0) * v
    return result

def createDataSet():
    dataSet = [[1, 1, 'yes'],
            [1, 1, 'yes'],
            [1, 0, 'no'],
            [0, 1, 'no'],
            [0, 1, 'no']]
    labels = ["test", "new"]
    return dataSet, labels

# split the date set with specified feature and value
def splitDateSet(dataset, feature, val):
    newset = []
    for data in dataset:
        if data[feature] == val:
            newset.append(data[:feature] + data[feature+1:])
    return newset

# choose the best feature
def chooseBestFeature(dataset):
    numfeature = len(dataset[0]) - 1
    entropy = computeEntropy(dataset)
    bestFeature = -1
    bestEntropy = 0.0
    lenDataSet = len(dataset)
    for i in range(numfeature):
        featureSet = set([e[i] for e in dataset])
        currEntropy = 0.0
        for f in featureSet:
            newSet = [data for data in dataset if data[i] == f]
            currEntropy += float(len(newSet)) / lenDataSet * computeEntropy(newSet)
        gain = entropy - currEntropy
        if gain > bestEntropy:
            bestEntropy = gain
            bestFeature = i
    return bestFeature

# construct Decision Tree
def constructDecisionTree(dataset, labelname):
    # if all belongs to one class, return the result
    labelVec = [d[-1] for d in dataset]
    if len([l for l in labelVec if l == labelVec[0]]) == len(labelVec):
        return labelVec[0] # all belongs to one class
    if len(labelVec) == 1:
        return labelVec[0]
    bestFeature = chooseBestFeature(dataset)
    bestFeatureLabelName = labelname[bestFeature]
    tree = {bestFeatureLabelName:{}}
    featureVal = set([d[bestFeature] for d in dataset])
    for f in featureVal:
        newLabel = labelname[:]
        del(newLabel[bestFeature])
        tree[bestFeatureLabelName][f] = constructDecisionTree(splitDateSet(dataset, bestFeature, f), newLabel)
    return tree

if __name__ == '__main__':
    myData, label = createDataSet()
    print computeEntropy(myData)
    print chooseBestFeature(myData)
    print constructDecisionTree(myData, label)

