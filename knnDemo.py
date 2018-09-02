'''
https://www.cnblogs.com/ybjourney/p/4702562.html
https://blog.csdn.net/weixin_38705903/article/details/79231551
KNN算法实现
'''

from numpy import *
import operator


# shape[0]和shape[1]分别代表行和列的长度
def knn(k, testData, trainData, labels):
    trainDataSize = trainData.shape[0]
    dif = tile(testData, (trainDataSize, 1)) - trainData
    sqdif = dif ** 2
    sumsqdif = sqdif.sum(axis=1)
    distance = sumsqdif ** 0.5
    sortdistance=argsort(distance)
    count={}
    for i in range(0,k):
        vote=labels[sortdistance[i]]
        count[vote]=count.get(vote,0)+1
    sortCount=sorted(count.items(),key=operator.itemgetter(1),reverse=True)
    return sortCount[0][0]