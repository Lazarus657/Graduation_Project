# Pytorch学习日志

## 2023-01-09

1.1张量的定义

```python
张量Tensor,numpy的ndarray，可以在GPU上加速
from __future__ import print_function
import torch

x=torch.empty(5,3)#无初始化的矩阵
x=torch.rand(5,3)#随机初始化的矩阵
x = torch.zeros(5,3,dtype=torch.long)#构造一个填满 0 且数据类型为 long 的矩阵：
x = torch.tensor([5.5,3])#直接从数据构造张量

x = x.new_ones(5, 3, dtype=torch.double) 
#或根据现有的 tensor 建立新的 tensor 。除非用户提供新的值，否则这些方法将重用输入张量的属性，例如 dtype 等：
x = torch.randn_like(x, dtype=torch.float)
 # 重载 dtype!
print(x.size())
#张量的形状
#输出结果是tuple？？
```

1.2张量运算

```python
#加法1
y = torch.rand(5, 3)
print(x+y)
#加法2
print(torch.add(x,y))
#加法3,给定一个输出张量作为参
result=torch,empty
torch.add(x,y,out=result)
print(result)
#加法4
#add x to y
#原位/原地操作（in-place）
y.add_(x)
print(y)
!!!注意：任何一个就地改变张量的操作后面都固定一个_,例如 x.copy_（y）， x.t_（）将更改x
```

##### 1.3 杂谈

```python
#1.可以使用像标准的 NumPy 一样的各种索引操作：
print(x[:, 1])
#输出
tensor([-0.6769,  0.7683, -0.5566,  0.3566, -0.6741])

#2.改变形状：如果想改变形状，可以使用 torch.view
#rand 和 randn 有啥区别
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions也就是说-1代表依靠另外的维度呗
print(x.size(), y.size(), z.size())
#输出
torch.Size([4, 4]) torch.Size([16]) torch.Size([2, 8])

#3.如果是仅包含一个元素的 tensor，可以使用 .item（） 来得到对应的 python 数值
x = torch.randn(1)
print(x)
print(x.item())
#输出
tensor([0.0445])
0.0445479191839695
```

##### 1.4和Numpy

```python
将一个 Torch 张量转换为一个 NumPy 数组是轻而易举的事情，反之亦然。

#1.Torch 张量和 NumPy数组将共享它们的底层内存位置，因此当一个改变时，另外也会改变。
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
#输出
tensor([1., 1., 1., 1., 1.])
[1. 1. 1. 1. 1.]

#2.看 NumPy 细分是如何改变里面的值的：
a.add_(1)
print(a)
print(b)
#输出
tensor([2., 2., 2., 2., 2.])
[2. 2. 2. 2. 2.]

#3.numpy数组转换成torch张量
import numpy as np
a=np.ones(5)
b=torch.from_numpy(a)
np.add(a,1,out=a)
print(a)
print(b)

#4.cuda上的张量
#张量可以使用 .to 方法移动到任何设备（device）上：
# 当GPU可用时,我们可以运行以下代码
# 我们将使用`torch.device`来将tensor移入和移出GPU
if torch.cuda.is_avaiable():
    device=torch.device("cuda")# a CUDA device object
    y=torch.ones_like(x,device=device)#device=cuda
    x=x.to(device)
    z=x+y
    print(z)
    print(z.to("cpu",torch.double)) 
    # `.to`也能在移动时改变dtype
CUDA 11.1.144
这段代码运行不下去，因为没有安装cuda查询版本，解决方法找淘宝

```

[cuda版本查询](https://blog.csdn.net/chenhepg/article/details/104653354?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-104653354-blog-46362277.pc_relevant_3mothn_strategy_recovery&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-104653354-blog-46362277.pc_relevant_3mothn_strategy_recovery&utm_relevant_index=2)

2023-01-09

##### 2.1 Autograd自动求导?? 线性回归代码示例

```python
from __future__ import print_function
import torch
import numpy as np
import torch.nn as nn
class LinearModel(nn.Module):
    def __init__(self,input_dim,output_dim):#所以说这里在继承什么
        super(LinearModel,self).__init__()
        self.linear=nn.Linear(input_dim,output_dim)#定义了一个全连接层
# 按间距中的绿色按钮以运行脚本。
    def forward(self,x):
        out=self.linear(x)
        return out
if __name__ == '__main__':
        x_value=[i for i in range(11)]
    x_train=np.array(x_value,dtype=np.float32)
    x_train=x_train.reshape(-1,1)
    x_train.shape#这一步是在转换列成行向量？？

    y_value=[2*i+1 for i in x_value]
    y_train=np.array(y_value,dtype=np.float32)
    y_train=y_train.reshape(-1,1)
    y_train.shape


    input_dim=1
    output_dim=1
    model=LinearModel(input_dim,output_dim)
    print(model)

    epochs=1000#迭代次数
    learning_rate=0.01#优化器
    optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)#优化器
    criterion=nn.MSELoss()#损失函数？

    for epoch in range(epochs):
        epoch+=1
        inputs=torch.from_numpy(x_train)
        labels=torch.from_numpy(y_train)
        #梯度清零
        optimizer.zero_grad()
        #前向传递
        outputs=model(inputs)
        #计算损失
        loss=criterion(outputs,labels)
        loss.backward()

        optimizer.step()
        if epoch %50 == 0:
			 print('epoch {},loss {}'.format(epoch,loss.item()))
```

```python
tensor有哪些形式
0-3
scalar
vector #向量特征不能  一维的
matrix 一般计算都是矩阵，通常是多维的
n-dimensional tensor
```

```
hub模块 调用别人训练好的模块
但是网路不好用
```

2023-02-10

```python
回顾知识：
1.python中的__call__函数
2.python中super函数
super（）是用来解决多重继承问题，直接用类名调用父类方法使用单继承没有问题
参考地址
https://blog.csdn.net/weixin_42105064/article/details/80151587
3.self的解释
    方法里面的self代表的是当前类的实例化后的对象 self不是只能叫self 别的也可以 但是规范来     说 都使用self
    class A:
        name=''
        def demo(self):
            print(self)
            print(id(self))
    a=A()
    a.demo()
    print(a)
    print(id(a))
    —————————————————————————————————————————————————————————————————————————————
    输出结果
    <__main__.A object at 0x000001EB44947640>
    2109979522624
    <__main__.A object at 0x000001EB44947640>
    2109979522624
------------------------------------------------------------------------------
这里是return的用法，return遇到之后直接返回和c语言那边一样
class Demo:
    name=''
    def speak(self):
        print("我是{}号楼".format(self.name))

    def myReturn(self):
        return self.name
        print(self.name)
d2=Demo()
d2.name='d2'
d2.speak()
d2Name=d2.myReturn()

4.__init__
双下划线声明该属性为私有属性，第一个参数必须是self
参考地址
https://blog.csdn.net/luzhan66/article/details/82822896
def __init__(self, width, height, depth):
        self.width = width
        self.height = height
        self.depth = depth
 2023-02-18       
```

2023-02-19

搭建神经网络对气温进行预测

偏置参数和权重层有什么关系，还有权重为什么要有128个





平方损失，衡量预估质量

训练数据来决定参数，权重和偏差

损失函数是x，y，w，b的一个，目标要损失函数最小



02-20

没心情看。。。

02-21

考研成绩出来了，很差



02-23

总结学习

1.神经网络的理论学习：线性回归是怎么实现的y=a*x+b(权重和偏置)

2.python类相关的，继承，self的含义和用途，_ _init_ _声明代表是私用函数，类定义下的函数一定要先写self

今天任务 第三章 线性回归完整实现

```python
读取数据集
1.构造数据集
    torch.normal(mean，方差，size)
    "一个权重向量，一个偏置标量" 
    reshape(-1,1)
    "转换成一行"
    reshape(1,-1)
    "转换成一列"
    feature,label
    "feature是特征向量，label是真实值"
2.读取数据集
	random.shuffle(indices)
    "列表数字打乱，但不生成新的列表"
    for i in range(0,num,batch_size):
        batch_indices=torch.tensor(indices[i:min(i+batch_size)])
     "这里的切片操作没看懂"
    yield features[batch_indices],labels[batch_indices]
3.初始化模型参数
	w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
	b = torch.zeros(1, requires_grad=True)
4.定义模型
	torch.matmul(X, w) + b
5.定义损失函数
 	return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
	"均方损失"
6.定义优化算法
	"不太懂，要再去看看视频"
	def sgd(params, lr, batch_size):  #@save
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
 7.训练
```

全国大学生机器人竞赛ROBOCON

中国工程机器人大赛

蓝桥杯竞赛

重庆邮电大学单片机竞赛

重庆邮电大学实控大赛

简历投递 中移互联，CVTE,比亚迪

02-24

学习总结

1.线性回归的流程 数据集生成，读取数据集，初始化模型参数，定义模型，定义损失函数，定义优化函数，训练

 语法方面reshape(-1,1) reshape(1,-1) 表示行向量 列向量

2.with函数和生成器yield

```python
with 语句适用于异常处理，封装了 try...except...finally
"1.原代码"
file = open('./test_runoob.txt', 'w')
try:
    file.write('hello world')
finally:
    file.close()

"2.使用with"
with open('./test_runoob.txt', 'w') as file:
    file.write('hello world !')

```

```

```

02-24下午

回归和分类

02-26下午

复习之前内容

学习新知识

回归问题和分类问题

独热编码用来解决分类问题

从回归到多类分类-

y=argmax [oi]()

权重的表达式o=Wx+b

王木头softmax，交叉熵

熵的定义 类似于 概率中期望的定义

KL散度是交叉熵减去p系统的熵 吉布斯散度表示kl散度一定大于等于0 交叉熵最小 损失最小

常用的损失函数、

02-27

损失函数：（各种误差的优缺点）

1.均方损失MES

2.绝对值损失函数

3.鲁棒误差

```python
fashion-mnist数据集
totensor函数
'数据从PIL格式或者ndarry格式转换成tensor，并且进行归一化处理'
random.randint(start,stop,size)
reshape,这里-1可以表示任何数字，一种模糊控制
```

02-28

```python
x=torch.tensor[0,2]
y_hat=torch.tensor([[0.1,0.3,0.6],[0.3,0.2,0.5]])
"numpy的高级索引"
```

03-01

```python
1.
keepdim=True
"保持矩阵维数不变进行求和"
x.sum(0,keepdim=True);x.sum(1,keepdim=True)
2.
'softmax的实现'
x_exp=torch.exp(x)
partition=x.exp.sum(1,keepdim,True)
return x_exp/partition
"实现softmax函数"
x=torch.normal(0,1,(2,5))
x_prob=softmax(x)
"实现softmax代码"


def net(x)
	return softmax(torch.matmul(x.reshape((-1,w.shape[0])),w)+b)
"定义模型，这里的w大小是多少解答w定义的权重矩阵是784*10"
```

```python
"定义损失函数"
"这里的预测函数是交叉熵预测"
y=torch.tensor([0,2])
y_hat=torch.tensor([0.1,0.3,0.6],[0.3,0.2,0.5])
y_hat[[0,1],y]

"这里是python的双重索引或者是numpy神奇索引"
def cross_entropy(y_hat,y)
	return -torch.log(y_hat[range(len(y_hat)),y])
cross_entropy(y_hat,y)

def accuracy(y_hat,y):
    if len(y_hat.shape) >1 and hat.shape[1]>1:
    "这里有点不懂"
        y_hat=y_hat.argmax(axis=1)

     "这里的len(y_hat)输出的是y_hat的行数，不能使用shape[0],因为list不能用int"
    "axis=1就是沿着列方向进行，找出每一行最大概率对应的下标，axis=0就是输出每一列最大值的索引（重点求索引）"
        cmp=y_hat.type(y.type)==y
    return float(cmp.type(y.dtype).sum())
"这里是将y_hat的数据类型转换成为y的数据类型"
"cmp是一个bool变量"
```

```python
accuracy(y_hat,y)/len(y)
"这里计算出来预测概率"


"评估任意模型的"
def evaluate_accuracy(net,data_iter):
    if isinstance(net,torch.nn.Module):
"isinstance()函数判断一个对象是否为一个已知的类型"
        net.eval()
    	metric.Accumulator(2)
        with torch.no_gard():
"with 关键字用于异常处理"         
"评估模式，输入后得出来的结果评估模型的准确率，不做反向传播"
		metric.add(accuracy(net(X),y),y.numel())   "得到两个数据，正确预测数，预测总数"
    	return metric[0]/metric[1]
```

```python
class Accmulator:
def __init__(self,n):
    self.data=[0.0]*n
    
def add(self,*args):
    self.data=[a+float(b) for a,b in zip(self.data,args)]

def reset(self)；
	self.data=[0.0]*len(self.data)
```

```python
*args
"发送一个非键值对的可变数量的参数列表给一个函数"
"表示任何多个无名参数，他是一个元组"
```

[Python中*args](https://blog.csdn.net/qq_45893319/article/details/122040866?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522167817660016800213063192%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=167817660016800213063192&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-122040866-null-null.142^v73^control_1,201^v4^add_ask,239^v2^insert_chatgpt&utm_term=args&spm=1018.2226.3001.4187)

```
python中的__getitem__的使用
```

训练

```python
def train_epoch_ch3(net,train_iter,loss,updater):
    if isinstance(net,torch.nn.Module):
        net.train()
     metric=Accmulator(3)
    for x,y in train_iter:
        y_hat=net(x)
        l=loss(y_hat,y)
        if                                                                                                          
```

```
softmax简洁
flatten将任意维度的tensor转换成2d的tensor
```

>pip install -r D:\Download\yolov8_tracking-master\requirements.txt -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

```text
-i https://pypi.tuna.tsinghua.edu.cn/simple
```

conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.1 -c pytorch

```
list
yolo v8的使用数据集
kitty
训练中的tensorboard
损失函数
追踪图片的数据，画出表来
```

2023-03-09

```
yolov8 学习
kitti数据集
yolov8学习日志
```

项目常用指令

```python
python track.py --source 222.mp4 --yolo-weights yolov8n.pt  --save-trajectories --save-vid

```

anaconda常用指令

```python
"1. 跳转到上一级目录"
cd..
"2. 跳转到根目录 "
cd /
"3.跳转到指定目录"
cd path
"4.查找现有环境"
conda env list
"5.克隆环境"
conda create -n 新环境名 --clone 旧环境名
"6.pip换源"
-i 

"7.安装pytorch包指令"
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.1 -c pytorch

```

yolov8学习文档

```
coco128训练

```

1.数据集的准备

论文的主要目录：



1.国内外研究

2.神经网络

3.软件标注

4.介绍YOLO算法

5.介绍实验过程

6.总结

```python
Annotations是数据集的xml 
jpegimages对xml进行划分，再转换成txt文件
split.py 随机将数据集进行划分 运行随机分配训练/验证/测试集图片
xmltotxt "这里将xml文件转化成txt文件"
创建自己的yaml数据集
设置自己的配置 yolo树下的default.yaml文件

trainval_percent = 0.9 # 训练验证集占整个数据集的比重（划分训练集和测试验证集）
train_percent = 0.9  # 训练集占整个训练验证集的比重（划分训练集和验证集
```

```
kitti数据集
object-2d
```

问题：tensorboard不能运行，好像可以，最后数据出来了

```
YOLO训练步骤
1.train val predict 训练 验证 预测

2.Segmentation detection

3.对象检测与图像分割(语义处理)Segmentation
https://towardsdatascience.com/what-is-the-difference-between-object-detection-and-image-segmentation-ee746a935cc1


```





03-11

妈的忘记保存了日



权重和偏置

超参数和过拟合，泛化的问题

欧式距离，曼哈顿距离还有KNN算法的一些局限性



线性分类器  其实是一种模板匹配算法 局限 每个类别的一类算法  

线性分类器函数形式 函数的权重是怎样得到的

线性分类器中的每行的都是代表图片中某个像素对分类结果的影响



损失函数 度量w好坏的一个方法 

几种损失函数 

x是每个像素点构成的数据集 y是希望算法检测出来的东西 我们称为标签或者目标

和优化函数

损失函数 多分类SVM   hings loss 铰链损失函数

![]()

横轴是训练样本真实分类的分数

y轴是损失函数

随着x值的增加，y损失函数线性下降，当通过某个阈值时候损失函数为0

其实就是正确类别的得分比其他类大于1以上，就不产生损失 

为什么要加上1  其实可以任意选 我们关心的分数的相对偏差

如果汽车的损失函数有变化，那么损失函数将不会有任何变化  

损失函数的最小值是0   

为什么损失函数是c-1呢？？？

```
使用231n搭建环境遇到的问题
python2转换到python3
在
from __future__ import print_function
import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt
中的from cs231n.data_utils import load_CIFAR10
这里由于是2016年的作业现在库文件还用的python2.7版本，新版本要在google colab里面跑，主要有两个问题print函数 xrange函数
```

快速转换代码

```
2to3.py -w transform.py
```

knn部分算法

```python
python pass语句 空语句 保持程序结构完整
该处的 pass 便是占据一个位置，因为如果定义一个空函数程序会报错，当你没有想好函数的内容是可以用 pass 填充，使程序可以正常运行。

def sample(n_samples):
    pass
```

KNN怎么读取label和数据 

```
range函数是一个可迭代的对象，不是列表
```

复习numpy

numpy.reshape(arr,newshape)

数组的切片操作要用到range,好像有点奇怪

#arr 是要修改的数组

#newshape 是新的类型

np.square()

np.dot()

```python
 def compute_distances_no_loops(self,X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        test_sum=np.sum(np.square(X),axis=1)
        train_sum=np.sum(np.square(self.X_train),axis=1)
        inner_product=np.dot(X,self.X_train.T)
        #这一步是什么意思
        dists=np.sqrt(-2*inner_product+test_sum.reshape(-1,1)+train_sum)
        return dists
```

```python
closest_y = self.y_train[np.argsort(dists[i])[0:k]]
#y_train存的是标签，这里按行进行处理，打印的测试集的预测结果
np.argsort()函数 从小到大排列序号
python中向量化的思想 one_loop no_loop

 X_train_folds.append(X_train[i * avg_size : (i+1) * avg_size])
怎样对数据进行拆分的

np.hstack将参数元组元素数组按照水平方向进行叠加
np.vstack将参数元组元素数组按照垂直方向进行叠加
```

 ![](Pytorch学习日志.assets/image-20230315220647931.png)

numpy的广播机制

margins为什么要清零 

w的值改变会影响L那么到底怎么样才是一个好的损失函数呢

正则化表达式 鼓励模型选择更简单的数据r

r的两种方法，减小参数，减小阶数 

SOFTMAX函数和交叉熵

```python
cs231n的应用
softmax，svm的差异
```

 YOLOV5的应用

![image-20230316211334714](Pytorch学习日志.assets/image-20230316211334714.png)

inference介绍

```python
parser.add_argument
import os 环境变量
```

![image-20230316212630376](Pytorch学习日志.assets/image-20230316212630376.png)

  权重文件在哪里存放

修改指定路径？ 

```python
defalut=640
conf-thres 修改显示的阈值，置信度

iou-thres 非最大值抑制 区域交集/区域并集 避免重复判断

device 设备

view img 显示图片

tips 不使用命令行 就可以直接运行的方式pycharm的右上角

_class 多目标检测的类别，具体的类别在yaml文件里面查询

augment 增强检测 

update 将网络模型不重要的部分给去掉 不用管

断点和debug 下一步 requirements参数设置为true 参数的解析
```

 ![image-20230317212707517](Pytorch学习日志.assets/image-20230317212707517.png)

如何训练yolov5（云端和本地） 

```python
workers可以先设置成0 线程开位0
 yaml是数据集设置
rectangular train
resume中断后 要设置文件地址 last.pt 断点继续训练
锚点/框 moautoahcor百度
evolve超参数净化
数据集参数yaml的选择
DDP
adam
1.ctrl+f 查询
2.issue 查询
save_period default=-1
```

云端GPU训练

 

pytorch--小土堆

```python
dir(pytorch)
help(pytorch.3.a)
```

如何加载数据

dataset和dataloader

dataset提供一种方式获取数据以及label

1. 获取每一个数据以及label
2. 告诉我们总共有多少个数据

dataloader对dataset进行打包，为后面的网络提供不同的数据形式

```python 
from torch.utils.data import Dataset
help(Dataset)

------------------------------------------
class Dataset(typing.Generic)
 |  Dataset(*args, **kwds)
 |  
 |  An abstract class representing a :class:`Dataset`.
 |  
 |  All datasets that represent a map from keys to data samples should subclass
 |  it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
 |  data sample for a given key. Subclasses could also optionally overwrite
 |  :meth:`__len__`, which is expected to return the size of the dataset by many
 |  :class:`~torch.utils.data.Sampler` implementations and the default options
 |  of :class:`~torch.utils.data.DataLoader`.
 |  
 |  .. note::
 |    :class:`~torch.utils.data.DataLoader` by default constructs a index
 |    sampler that yields integral indices.  To make it work with a map-style
 |    dataset with non-integral indices/keys, a custom sampler must be provided.
 |  
 |  Method resolution order:
 |      Dataset
 |      typing.Generic
 |      builtins.object
 |  
 |  Methods defined here:
 |  
 |  __add__(self, other: 'Dataset[T_co]') -> 'ConcatDataset[T_co]'
 |  
 |  __getitem__(self, index) -> +T_co
 |  
 |  ----------------------------------------------------------------------
 |  Data descriptors defined here:
 |  
 |  __dict__
 |      dictionary for instance variables (if defined)
 |  
 |  __weakref__
 |      list of weak references to the object (if defined)
 |  
 |  ----------------------------------------------------------------------
 |  Data and other attributes defined here:
 |  
 |  __orig_bases__ = (typing.Generic[+T_co],)
 |  
 |  __parameters__ = (+T_co,)
 |  
 |  ----------------------------------------------------------------------
 |  Class methods inherited from typing.Generic:
 |  
 |  __class_getitem__(params) from builtins.type
 |  
 |  __init_subclass__(*args, **kwargs) from builtins.type
 |      This method is called when a class is subclassed.
 |      
 |      The default implementation does nothing. It may be
 |      overridden to extend subclasses.
 |  
 |  ----------------------------------------------------------------------
 |  Static methods inherited from typing.Generic:
 |  
 |  __new__(cls, *args, **kwds)
 |      Create and return a new object.  See help(type) for accurate signature.


进程已结束，退出代码为 0

```

```python
getitem
from PIL import Image
img_path=""
#这里要使用双引号引用地址否则会被认为是转义字符
#或者加r防止转义
img=Image.open(img_path)

"获取地址的索引"
import os
img_path_list=os.listdir(dir_path)

def __init__(self.root_dir,label_dir):
	"一个函数变量不能传递给另外一个函数，self相当于指定了一个类中的全局变量" 
    self.root_dir=root_dir
```

```python
tensorboard的使用
from torch.utils.tensorboard import SummaryWriter

writer=SummaryWriter("logs")

writer.add_image()
writer.add_scale()
#图标的标题 数值 
```

