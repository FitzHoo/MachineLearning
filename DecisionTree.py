import matplotlib.pyplot as plt
from math import log
import os

os.chdir("/Users/huliangyong/Docs/MachineLearning/machinelearninginaction/Ch03")


# 信息熵的最小值为0，最大值为log(2,|Y|)，熵越小纯度越高
#
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0
    for key in labelCounts:
        prob = labelCounts[key] / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def creatDataSet():
    dataSet = [[1, 1, "yes"],
               [1, 1, "yes"],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, "no"]]
    labels = ["no surfacing", "flippers"]
    return dataSet, labels


# Calculate the entropy
entropy = calcShannonEnt(creatDataSet()[0])


def splitDataSet(dataSet, axis, value):
    """
    按照给定特征划分数据集
    :param dataSet: 待划分的数据集
    :param axis: 划分数据集的特征
    :param value: 特征的返回值
    :return:
    """
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[: axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)

    return retDataSet


myData, labels = creatDataSet()
splitDataSet(myData, axis=0, value=1)


def featureSelection(dataSet):
    numFeatures = len(dataSet[0]) - 1  # feature numbers
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0
    bestFeature = -1
    for i in range(numFeatures):
        featureList = [s[i] for s in dataSet]  # the ith feature list
        uniqueVals = set(featureList)
        newEntropy = 0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / len(dataSet)
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


from operator import itemgetter


# 返回出现次数最多的分类名称
def majorityCount(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=itemgetter(1), reverse=True)

    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    """
    决策树递归函数
    :param dataSet: 数据集
    :param labels: 标签列表
    :return: 返回嵌套很多代表叶子节点信息的字典数据
    """
    classList = [s[-1] for s in dataSet]  # 叶子节点

    # 递归停止条件：1）所有的类标签完全相同，直接返回该标签
    #             2）使用完所有的特征，仍不能将数据集划分成仅包含唯一类别的分组，此时使用出现次数最多的类别作为返回值
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:  # 仅有一个特征
        return majorityCount(classList)

    bestFeature = featureSelection(dataSet)  # 最优特征的位置
    bestFeatureLabel = labels[bestFeature]

    myTree = {bestFeatureLabel: {}}
    del labels[bestFeature]  #
    featureValues = [s[bestFeature] for s in dataSet]

    uniqueValues = set(featureValues)
    for value in uniqueValues:
        subLabels = labels[:]
        myTree[bestFeatureLabel][value] = createTree(splitDataSet(dataSet, bestFeature, value), subLabels)

    return myTree


# 使用文本注释绘制树节点
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords="axes fraction", xytext=centerPt, arrowprops=arrow_args,
                            textcoords="axes fraction", va="center", ha="center", bbox=nodeType)


def createPlot(inTree):
    fig = plt.figure(1, facecolor="white")
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = getNumLeafs(inTree)
    plotTree.totalD = getTreeDepth(inTree)
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1
    plotTree(inTree, (0.5, 1.0), '')
    # plotNode("a decision node", (0.5, 0.1), (0.1, 0.5), decisionNode)
    # plotNode("a leaf node", (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()


def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == "dict":
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if isinstance(secondDict[key], dict):
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


def retrieveTree(i):
    listOfTrees = [{"no surfacing": {0: "no", 1: {"flippers": {0: "no", 1: "yes"}}}},
                   {"no surfacing": {0: "no", 1: {"flipper": {0: {"head": {0: "no", 1: "yes"}}, 1: "no"}}}}]
    return listOfTrees[i]


def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)


def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + numLeafs) / 2 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if isinstance(secondDict[key], dict):
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1 / plotTree.totalD


# 决策树分类
def classify(inputTree, featureLabels, testVec):
    firstStr = [*inputTree][0]
    secondDict = inputTree[firstStr]
    featureIndex = featureLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featureIndex] == key:
            if isinstance(secondDict[key], dict):
                classLabel = classify(secondDict[key], featureLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


def storeTree(inputTree, filename):
    import pickle
    with open(filename, "w") as fw:
        pickle.dump(inputTree, fw)


def grabTree(filename):
    import pickle
    with open(filename, "r") as fr:
        result = pickle.load(filename)
    return result


#
# TEST FILE
#
with open("lenses.txt", mode="r") as fr:
    lenses = [s.strip().split("\t") for s in fr.readlines()]
lensesLabels = ["age", "prescript", "astigmatic", "tearRate"]
lensesTree = createTree(lenses, lensesLabels)
createPlot(lensesTree)
