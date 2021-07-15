import numpy as np
def sigmoid(x,deriv=False): #定义求导激活函数,默认false不计算导数
    if(deriv == True):      #等于true时反向传播求导
        return x*(1-x)       #此时输入样本x经过了激活函数sigmoid，变成了1/(1+e^(-x))，而导数等于sigmoid（1-sigmoid）
    return 1/(1+np.exp(-x) )#正常情况（false）下前向传播的值

x=np.array([[0,0,1],         #自己造的数据，每个数据3个特征
            [0,1,1],
            [1,0,1],
            [1,1,1],
            [0,0,1]]
)
print(x.shape)               #输出x的形状是个5sample3特征值的元素


y=np.array([[0],[1],[1],[0],[0]])   #label标签y,[0]代表属于0类
print(y.shape)

np.random.seed(1)                   #指定随机值1，生成一组随机值

#构造连接不同层之间的权重函数w0,w1
w0=2*np.random.random((3,4))-1          #随机生成3行4列权重函数，并对其进行正则化，x*w0为[5,3]*[3,4]，4是由第一个隐含层有四个神经元确定的
w1=2*np.random.random((4,1))-1          #连接隐含层与输出层的权重函数，输出层只有一个神经元，值为0或1，以进行分类
print(w0)

# ##下构造神经网络与迭代计算
# 神经网络共有三层（L0,L1,L2）
# L2输出0或1并与label值y进行比较，更新权重
for j in range(60000):                   #迭代60000次
    L0 = x
    L1 = sigmoid(np.dot(L0, w0))             #定义L1层，np.dot是矩阵乘,在走完第一层后还要进行激活处理，故在np.dot(L0,w0)外加一层sigmoid
    L2 = sigmoid(np.dot(L1, w1))             #sigmoid将输入值压缩为0，1上
    L2_error = y-L2                     #L2层输出和y之间的误差
    if(j%10000)== 0:
        print ('Error'+str(np.mean(np.abs(L2_error))))  #打印当前的误差值

# 前向跑完一遍神经网络后，进行反向传播
    L2_delta = L2_error * sigmoid(L2,deriv = True)      #点乘[5,1]*[5,1]    #此处的L2_error作为更新的权重项，错的越多，更新力度越大，此行代码后值被传到L2层和sigmoid层之间
#再向前传
    L1_error = L2_delta.dot(w1.T)           #L2_delta是5×1矩阵，w1是4×1矩阵，要对w1转置。对矩阵求导
    L1_delta = L1_error * sigmoid(L1, deriv=True)

#下更新权重函数
    w1 += L1.T.dot(L2_delta)                #L1.T是对L1*w1对w1求偏导得到的，w1对L2_delta（L2层的误差，每个样本错了多少）的贡献
    w0 += L0.T.dot(L1_delta)
#for循环完成
#神经网络完成