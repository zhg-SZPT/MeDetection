
from torch.utils.data import Dataset,DataLoader

class MyDataSub(Dataset):
    #task_names从外界传入为了区分maml训练集和测试任务
    def __init__(self, data,lable,imgs_name,flag):
        self.data=data
        self.lable=lable
        self.imgs_name=imgs_name
        self.flag=flag
    def __getitem__(self, item):
        if(self.flag=="train"):
            return self.data[item],self.lable[item]
        if(self.flag=="test"):
            return self.data[item], self.lable[item],self.imgs_name[item]
    def __len__(self):
        return self.data.shape[0]