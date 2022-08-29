import  torch
from    torch.utils.data import TensorDataset, DataLoader
import  numpy as np

# 交叉验证函数
def kfold( k, i, data, label):
    # 位置信息
    set_len = len(data)
    pos = list(range(0, set_len))
    patch = set_len // k
    testSet_pos = pos[i * patch:(i + 1) * patch]
    trainSet_pos = pos.copy()
    for vl in testSet_pos:
        trainSet_pos.remove(vl)
    # 测试集
    # x
    testSet_x = torch.utils.data.Subset(data, testSet_pos)
    testSet_x = testSet_x.dataset[testSet_pos]
    # y
    testSet_y = torch.utils.data.Subset(label, testSet_pos)
    testSet_y = testSet_y.dataset[testSet_pos]
    # 训练集
    # x
    trainSet_x = torch.utils.data.Subset(data, trainSet_pos)
    trainSet_x = trainSet_x.dataset[trainSet_pos]
    # y
    trainSet_y = torch.utils.data.Subset(label, trainSet_pos)
    trainSet_y = trainSet_y.dataset[trainSet_pos]
    return (trainSet_x, trainSet_y, testSet_x, testSet_y)

def getZeroSampler(data, label):
    # selected_imgs_idx = np.random.choice(zeroSampler_len, batchsize, False)
    # np.random.shuffle(selected_imgs_idx)
    l = data.shape[0]
    zero_pos=[]
    for i in range(0, l):
        if (label[i] == 0):
            zero_pos.append(i)
    zeroSampler = torch.utils.data.Subset(data, zero_pos)
    zeroSampler = zeroSampler.dataset[zero_pos]
    zeroLabel = torch.utils.data.Subset(label, zero_pos)
    zeroLabel= zeroLabel.dataset[zero_pos]
    return zeroSampler,zeroLabel
def getOneSampler(data, label):
    # selected_imgs_idx = np.random.choice(zeroSampler_len, batchsize, False)
    # np.random.shuffle(selected_imgs_idx)
    l = data.shape[0]
    one_pos=[]
    for i in range(0, l):
        if (label[i] == 1):
            one_pos.append(i)
    oneSampler = torch.utils.data.Subset(data, one_pos)
    oneSampler = oneSampler.dataset[one_pos]
    oneLabel = torch.utils.data.Subset(label, one_pos)
    oneLabel= oneLabel.dataset[one_pos]
    return oneSampler,oneLabel

def balanceSet(data,label):
    zeroSampler,zeroLabel=getZeroSampler(data,label);
    oneSampler,oneLabel=getOneSampler(data,label);
    oneSize=oneSampler.shape[0]
    zeroSampler=zeroSampler[0:oneSize]
    data_balance=torch.vstack((zeroSampler,oneSampler))
    label_balance=torch.hstack((zeroLabel,oneLabel))
    for i in range(0,4):
        data_balance,label_balance=Nrand1(data_balance,label_balance)
    return data_balance,label_balance

def getBatchZeroSampler(data, batchsize):
    selected_imgs_idx = np.random.choice(data.shape[0], batchsize, False)
    np.random.shuffle(selected_imgs_idx)
    zeroSampler = torch.utils.data.Subset(data, selected_imgs_idx)
    zeroSampler = zeroSampler.dataset[selected_imgs_idx]
    return zeroSampler
def getBatchSampler(data,label, batchsize):
    selected_imgs_idx = np.random.choice(data.shape[0], batchsize, False)
    np.random.shuffle(selected_imgs_idx)
    Sampler = torch.utils.data.Subset(data, selected_imgs_idx)
    Sampler = Sampler.dataset[selected_imgs_idx]
    s_label = torch.utils.data.Subset(label, selected_imgs_idx)
    s_label = s_label.dataset[selected_imgs_idx]
    return Sampler,s_label
 #将数据随机打乱
def Nrand1(data_x,data_y):
    selected_imgs_idx = np.random.choice(len(data_x), len(data_x), False)
    np.random.shuffle(selected_imgs_idx)
    num,c,h,w=data_x.size()
    t_data_x =  torch.FloatTensor(num,c,h,w)
    t_data_y = np.zeros((num), dtype=np.int)
    i = 0
    for t in selected_imgs_idx:
        t_data_x[i] = data_x[t]
        t_data_y[i] = data_y[t]
        i=i+1
    return (t_data_x,torch.LongTensor(t_data_y))