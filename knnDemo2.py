'''

knn算法实现手写识别
    1 将图片转为固定宽高并转换为文本
    2 将文本转换为数组
    3 将所有文本和结果都加载到分别加载到各自的数组中，从而建立训练集
    4 调用测试数据并调用knn算法，验证结果

'''
from PIL import Image
from os import listdir
from numpy import *
import operator


# 1. 图片处理将图片转为固定宽高，比如32*32 然后再转为文本
def img2txt(fileName):
    img = Image.open(fileName)
    file = open(fileName.split(".")[0] + ".txt", "a")
    # 分别得到图片宽高，单位为像素
    width = img.size[0]
    height = img.size[1]
    for i in range(0, width):
        for j in range(0, height):
            color = img.getpixel((i, j))
            colorTotal = color[0] + color[1] + color[2]
            print(color)
            if (600<colorTotal <= 765):
                file.write("0")
            else:
                file.write("1")
        file.write("\n")
    file.close()


# 2. 将文本转换为数组
def data2Array(file):
    arr = []
    fileContent = open(file)
    for i in range(0, 32):
        line = fileContent.readline()
        for j in range(0, 32):
            arr.append(int(line[j]))
    return arr


# 3.将所有文本和结果都加载到分别加载到各自的数组中，从而建立训练集
def trainData():
    labels = []
    trainFile = listdir("HandTrainingData")
    num = len(trainFile)
    # 长度1024（列），每一行存储一个文件
    # 用一个数组存储所有训练数据，行：文件总数 列：32*32
    trainArr = zeros((num, 1024))
    for i in range(0, num):
        fileName = trainFile[i]
        label = fileName.split('_')[0]
        labels.append(label)
        # https://blog.csdn.net/sinat_34474705/article/details/74458605 查看numpy的对array的切片赋值
        trainArr[i, :] = data2Array("HandTrainingData/" + fileName)
    return trainArr, labels


# 4.得到测试数据调用KNN算法去测试，是否能正确识别
def testData():
    trainArr, labels = trainData()
    testList = listdir("HandTestData")
    tnum = len(testList)
    for i in range(0, tnum):
        fileName = testList[i]
        testArr = data2Array("HandTestData/" + fileName)
        result = knn(3, testArr, trainArr, labels)
        print(result)


# KNN算法
def knn(k, testData, trainData, labels):
    trainDataSize = trainData.shape[0]
    dif = tile(testData, (trainDataSize, 1)) - trainData
    sqdif = dif ** 2
    sumsqdif = sqdif.sum(axis=1)
    distance = sumsqdif ** 0.5
    sortdistance = argsort(distance)
    count = {}
    for i in range(0, k):
        vote = labels[sortdistance[i]]
        count[vote] = count.get(vote, 0) + 1
    sortCount = sorted(count.items(), key=operator.itemgetter(1), reverse=True)
    return sortCount[0][0]


# 验证集失败，究其原因是因为对像素点的采集不够好,而且像素点不高，因此不能够knn算法不能够正确识别

if __name__ == '__main__':
    testData()
    # img2txt("9.png")