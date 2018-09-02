[参考网站一](https://www.cnblogs.com/ybjourney/p/4702562.html)
[参考网站二](https://blog.csdn.net/weixin_38705903/article/details/79231551)

```python
from numpy import *
train=array([[1,2],[3,4],[5,6],[7,8]])
test=array([3,3])
# 欧氏距离计算方法[(x2-x1),(y2-y1)]
diff=tile(test,(train.shape[0],1))-train
diff
```




    array([[ 2,  1],
           [ 0, -1],
           [-2, -3],
           [-4, -5]])




```python
# 欧氏距离计算方法[(x2-x1),(y2-y1)]**2
```


```python
sqdiff=diff**2
sqdiff
```




    array([[ 4,  1],
           [ 0,  1],
           [ 4,  9],
           [16, 25]], dtype=int32)




```python
# 欧氏距离计算方法△x**2+△y**2
```


```python
sum(sqdiff,axis=1)
```




    array([ 5,  1, 13, 41], dtype=int32)




```python
# 欧氏距离计算方法(△x**2+△y**2)**0.5
```


```python
dist=sum(sqdiff,axis=1)**0.5
```


```python
dist
```




    array([2.23606798, 1.        , 3.60555128, 6.40312424])




```python
sortedDistIndex=argsort(dist)
```


```python
# 对距离进行排序，结果以序号表示
```


```python
sortedDistIndex
```




    array([1, 0, 2, 3], dtype=int64)




```python
sortedDistIndex[0]
```




    1




```python
sortedDistIndex[2]
```




    2




```python
# 对类型进行归类 投票方式
```


```python
type=['A','A','B','B']
classCount={}
for i in range(0,3):
    vote=type[sortedDistIndex[i]]
    classCount[vote]=classCount.get(vote,0)+1
classCount
```




    {'A': 2, 'B': 1}




```python
# 投票方式最多者为最终结果
```


```python
import operator
sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
```




    [('A', 2), ('B', 1)]




```python
count=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
count
```




    [('A', 2), ('B', 1)]




```python
count[0]
```




    ('A', 2)




```python
count[0][0]
```




    'A'


