import os
import PIL.BmpImagePlugin
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import transforms
import numpy as np
from PIL import Image
import math

class MyData(Dataset):
    #task_names从外界传入为了区分maml训练集和测试任务
    def __init__(self, root,task_names, mode, batchsz, n_way, k_shot, k_query, resize, startidx=0):
        self.batchsz = batchsz
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        self.resize = resize
        self.startidx = startidx
        print('shuffle DB :%s, b:%d, %d-way, %d-shot, %d-query, resize:%d' % (
        mode, batchsz, n_way, k_shot, k_query, resize))
        self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                                 transforms.Resize((self.resize, self.resize)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])
        self.path = root
        self.data_x_support = {}
        self.data_y_support = {}
        self.data_x_support_0 = {}
        self.data_x_query = {}
        self.data_y_query = {}

        self.batchsz=len(task_names)
        task_names = [self.path + t for t in task_names]
        i = 0
        for task_name in task_names:
            class_names=os.listdir(task_name)
            class_names = [task_name + "/"+t for t in class_names]
            j=0
            s_data_x=[]
            s_data_y=[]
            q_data_x=[]
            q_data_y=[]
            #循环每一个类
            for class_name in class_names:
                #获取标签
                il = class_name.rindex('/')
                label=int(class_name[il + 1:])
                pic_names = os.listdir(class_name)
                #产生图片地址
                pic_names = [class_name+ "/"+t for t in pic_names]
                #除掉ini文件
                pic_names=self.ignore_ini(pic_names)
                #产生标签
                pic_y = [label for i in range(0, len(pic_names))]
                # 从每一类中挑选支持集和测试机
                    #按照输入的比例计算每个类支持集和查询集该多少个
                pic_names_num=len(pic_names)
                pic_rate= self.k_shot/(self.k_query+self.k_shot)
                k_shot_num=math.floor(pic_names_num*pic_rate)
                k_query_num = pic_names_num-k_shot_num
                (s_data_x_item, s_data_y_item, q_data_x_item, q_data_y_item) = self.Nrand(pic_names, pic_y, k_shot_num, k_query_num)
                if(label==0):
                    s_data_0_x  = s_data_x_item
                    s_data_0_y  = s_data_y_item
                if(label==1):
                    s_data_1_x = s_data_x_item
                    s_data_1_y = s_data_y_item
                q_data_x.extend(q_data_x_item)
                q_data_y.extend(q_data_y_item)
                j = j + 1
            #使数据平衡
            s_data_1_len=len(s_data_1_x)
            #取与异常样本相同数量的正常样本
            s_data_0_x_copy=s_data_0_x.copy()
            #截断操作会改变原来的值
            s_data_0_x = s_data_0_x[0:s_data_1_len]
            s_data_0_y = s_data_0_y[0:s_data_1_len]

            s_data_0_x.extend(s_data_1_x)
            s_data_0_y.extend(s_data_1_y)

            s_data_x=s_data_0_x
            s_data_y=s_data_0_y
            (s_data_x,s_data_y)=self.Nrand1(s_data_x,s_data_y)
            (q_data_x, q_data_y) = self.Nrand1(q_data_x, q_data_y)
            self.data_x_support[i] = s_data_x
            self.data_y_support[i] = s_data_y
            self.data_x_support_0[i] = s_data_0_x_copy
            self.data_x_query[i] = q_data_x
            self.data_y_query[i] = q_data_y
            i = i + 1
        self.data_x_support = list(self.data_x_support.values())
        self.data_y_support = list(self.data_y_support.values())
        self.data_x_support_0 = list(self.data_x_support_0.values())
        self.data_x_query = list(self.data_x_query.values())
        self.data_y_query = list(self.data_y_query.values())
    def __getitem__(self, index):
        #支持集和查询集数量
        bs=40
        bq=40
        #初始化
        data_x_name=[]
        data_y_name=[]
        data_x_0_name=[]
        t_support_x_name = list(self.data_x_support[index].values())
        t_support_y_name = list(self.data_y_support[index].values())
        t_support_x_name_0 = self.data_x_support_0[index]
        t_support_y_name_0=[0 for _ in range(0,len(t_support_x_name_0))]
        t_query_x_name = list(self.data_x_query[index].values())
        t_query_y_name = list(self.data_y_query[index].values())
        #对支持集随机选取N张图片
        support_x_name,support_y_name=self.randomSelect(t_support_x_name,t_support_y_name,bs)
        support_x_name_0,_=self.randomSelect(t_support_x_name_0,t_support_y_name_0,bs)
        #对支持集随机选取N张图片
        query_x_name,query_y_name=self.randomSelect(t_query_x_name,t_query_y_name,bq)
        query_x_name_0, _ = self.randomSelect(t_support_x_name_0, t_support_y_name_0, bq)
        #将两个集合合在一块
        data_x_name.extend(support_x_name)
        data_x_name.extend(query_x_name)
        data_y_name.extend(support_y_name)
        data_y_name.extend(query_y_name)
        data_x_0_name.extend(support_x_name_0)
        data_x_0_name.extend(query_x_name_0)

        self.setsz=len(support_x_name)
        self.setsz_0=len(support_x_name_0)
        self.querysz=len(query_x_name)
        self.querysz_0 = len(query_x_name_0)
        self.sz = len(data_x_name)
        self.sz_0 = len(data_x_0_name)
        support_x = torch.FloatTensor(self.setsz, 3, self.resize, self.resize)
        support_x_0 = torch.FloatTensor(self.setsz_0, 3, self.resize, self.resize)
        support_y = np.zeros((self.setsz), dtype=np.int)
        query_x = torch.FloatTensor(self.querysz, 3, self.resize, self.resize)
        query_y = np.zeros((self.querysz), dtype=np.int)
        query_x_0 = torch.FloatTensor(self.querysz_0, 3, self.resize, self.resize)
        data_x = torch.FloatTensor(self.sz, 3, self.resize, self.resize)
        data_x_0 = torch.FloatTensor(self.sz_0, 3, self.resize, self.resize)
        data_y = np.zeros((self.sz), dtype=np.int)
        #支持集转换为图片
        i=0;
        for path in support_x_name:
            img=self.transform(path)
            support_x[i]=img
            support_y[i]=support_y_name[i]
            i=i+1
        i=0;
        for path in support_x_name_0:
            img=self.transform(path)
            support_x_0[i]=img
            i=i+1
        #查询集转化为图片
        i = 0;
        for path in query_x_name:
            img = self.transform(path)
            query_x[i] = img
            query_y[i] = query_y_name[i]
            i = i + 1
        i = 0;
        for path in query_x_name_0:
            img = self.transform(path)
            query_x_0[i] = img
            i = i + 1
        i = 0;
        for path in data_x_name:
            img = self.transform(path)
            data_x[i] = img
            data_y[i] = data_y_name[i]
            i = i + 1
        i = 0;
        for path in data_x_0_name:
            img = self.transform(path)
            data_x_0[i] = img
            i = i + 1
        return data_x, torch.LongTensor(data_y), data_x_0
    def __len__(self):
        return self.batchsz
    #因为磁瓦数据集老是有ini文件，故在此删除它
    def ignore_ini(self,picnames):
        try:
          picnames.remove("desktop.ini")
        except:
            pass
        return picnames
    #从一个任务的一类数据集中随机支持集和测试集合
    def Nrand(self,data_x,data_y, kshot,kquery):
        selected_imgs_idx = np.random.choice(len(data_x), kshot + kquery, False)
        np.random.shuffle(selected_imgs_idx)
        indexDtrain = np.array(selected_imgs_idx[:kshot])
        indexDtest = np.array(selected_imgs_idx[kshot:])
        s_data_x = {}
        s_data_y = {}
        q_data_x = {}
        q_data_y = {}
        i = 0
        for t in indexDtrain:
            s_data_x[i] = data_x[t]
            s_data_y[i] = data_y[t]
            i=i+1
        i = 0
        for t in indexDtest:
            q_data_x[i] = data_x[t]
            q_data_y[i] = data_y[t]
            i=i+1
        return (list(s_data_x.values()),list(s_data_y.values()),list(q_data_x.values()),list(q_data_y.values()))
    #将数据随机打乱
    def Nrand1(self,data_x,data_y):
        selected_imgs_idx = np.random.choice(len(data_x), len(data_x), False)
        np.random.shuffle(selected_imgs_idx)
        t_data_x = {}
        t_data_y = {}
        i = 0
        for t in selected_imgs_idx:
            t_data_x[i] = data_x[t]
            t_data_y[i] = data_y[t]
            i=i+1
        return (t_data_x,t_data_y)
    #随机选取一张图片
    def randomSelectOne(self,data_x,data_y):
        selected_imgs_idx = np.random.choice(len(data_x), 1, False)
        np.random.shuffle(selected_imgs_idx)
        t_data_x = ""
        t_data_y = 0
        i = 0
        for t in selected_imgs_idx:
            t_data_x = data_x[t]
            t_data_y = data_y[t]
            i=i+1
        return (t_data_x,t_data_y)
    #随机选取N张图片
    def randomSelect(self,data_x,data_y,bachsize):
        selected_imgs_idx = np.random.choice(len(data_x), 1, False)
        np.random.shuffle(selected_imgs_idx)
        t_data_x = []
        t_data_y = []
        for i in range(0,bachsize):
            x,y = self.randomSelectOne(data_x,data_y)
            t_data_x.append(x)
            t_data_y.append(y)
        return (t_data_x,t_data_y)
if __name__ == '__main__':
    mini = MyData("./非平衡数据/", mode='train', n_way=2, k_shot=75, k_query=15, batchsz=13, resize=84)
#    data = MyData("./目标任务/", mode='train', n_way=2, k_shot=75, k_query=13, batchsz=13, resize=84)
    db = DataLoader(mini, 2, shuffle=True, num_workers=0, pin_memory=True)
    device = torch.device('cuda')
    i=0
    for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):
        x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
        print(x_spt.size(), y_spt.size(), x_qry.size(), y_spt.size())
        i=i+1
    print(i)





