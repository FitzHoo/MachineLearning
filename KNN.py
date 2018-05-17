import numpy as np
import matplotlib.pyplot as plt

from operator import itemgetter

import os
os.chdir("/Users/huliangyong/Docs/MachineLearning/machinelearninginaction/Ch02")


def classifier(initX, dataSet, labels, k):
    diffMat = initX - dataSet
    distance = np.square(diffMat).sum(axis=1)
    sortedDistIndicies = distance.argsort()
    classCount = {}
    for i in range(k):
        X_label = labels[sortedDistIndicies[i]]
        classCount[X_label] = classCount.get(X_label, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=itemgetter(1), reverse=True)

    return sortedClassCount[0][0]


def file2matrix(filename):
    with open(filename, mode="r") as fr:
        lines = fr.readlines()
        numberOfLines = len(lines)
        returnMat = np.zeros((numberOfLines, 3))
        classLabelVector = []
        index = 0
        mapping = {'largeDoses': 3, "smallDoses": 2, "didntLike": 1}
        for line in lines:
            line = line.strip()
            listFromLine = line.split("\t")
            returnMat[index, :] = listFromLine[0: 3]
            classLabelVector.append(mapping[listFromLine[-1]])
            index += 1

    return returnMat, classLabelVector


# Scatter Plot
filename = "datingTestSet.txt"
datingDataMat, datingLables = file2matrix(filename)
fig, ax = plt.subplots()
ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15*np.array(datingLables), 15*np.array(datingLables))
# ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], c=["r", "g", "b"])
ax.set_xlabel = "flight distance"
ax.set_ylabel = "game consuming time"
# ax.legend({'largeDoses': 3, "smallDoses": 2, "didntLike": 1})
plt.show()


# Normalization
def normalize(dataSet):
    normData = (dataSet - dataSet.min(axis=0)) / (dataSet.max(axis=0) - dataSet.min(axis=0))
    ranges = dataSet.max(axis=0) - dataSet.min(axis=0)
    return normData, ranges, dataSet.min(axis=0)



# classifier effect test
def datingClassTest():
    testRatio = 0.10
    datingDataMat, datingLables = file2matrix("datingTestSet.txt")
    normMat, ranges, minVals = normalize(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * testRatio)
    errorCount = 0
    for i in range(numTestVecs):
        classifierResult = classifier(normMat[i, :], normMat[numTestVecs: m, :],
                                      datingLables[numTestVecs: m], 3)
        if (classifierResult != datingLables[i]):
            errorCount += 1
            print("The classifier came back with {}, the real answer is {}".format(classifierResult, datingLables[i]))
    print("The total error rate is:{}".format(errorCount / numTestVecs))


def imag2vector(filename):
    returnVect = np.zeros((1, 1024))
    with open(filename, mode="r") as fr:
        for i in range(32):
            lineStr = fr.readline()
            for j in range(32):
                returnVect[0, 32*i+j] = int(lineStr[j])
        return returnVect

testVector = imag2vector("./digits/testDigits/0_13.txt")


from os import listdir
def handwritingClassTest():
    labels = []
    trainingFileList = listdir("./digits/trainingDigits")
    m = len(trainingFileList)
    trainingMat = np.zeros((m , 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split(".")[0]
        classNumStr = int(fileStr.split("_")[0])
        labels.append(classNumStr)
        trainingMat[i, :] = imag2vector("./digits/trainingDigits/{}".format(fileNameStr))

    testFileList = listdir("./digits/testDigits")
    errorCount = 0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split(".")[0]
        classNumStr = int(fileStr.split("_")[0])
        vectorUnderTest = imag2vector("./digits/testDigits/{}".format(fileNameStr))
        classifierResult = classifier(vectorUnderTest, trainingMat, labels, 3)

        if classifierResult != classNumStr:
            errorCount += 1
            print("The classifier came back with {}, the real answer is {}".format(classifierResult, classNumStr))
    print("Total number of errors is {} and error rate is {}".format(errorCount, errorCount/mTest))


