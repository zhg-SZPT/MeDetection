import numpy as np
import torch
from torch import nn
from util.ours import Ours
import learn2learn as l2l
from torch.nn import functional as F
from util.spp_layer import  spatial_pyramid_pool
from util.MyMamlData import MyData
from torch.utils.data import  DataLoader
import os
import  argparse

def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(
               batch_data,
               learner,
               features,
               loss,
               adaptation_steps,
               device
              ):
    #前向传播过程
    data, labels,data_0 = batch_data
    data, labels ,data_0= data.to(device), labels.to(device),data_0.to(device)
    output1 = features[0](data)
    output2 = features[0](data_0)
    output = torch.abs(output1 - output2)
    output = F.interpolate(output, size=[256, 256], mode="nearest")
    feature = features[1](output)
    feature = features[2](feature)
    feature = features[3](feature)
    feature = features[4](feature)
    x = features[5](feature)
    data = spatial_pyramid_pool(x, x.size(0), [x.size(2), x.size(3)], [4, 2, 1])
    #将数据分为支持集和查询集（在数据集里其实已经分好，前一半是支持后一半是查询）
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    l=data.shape[0]
    a=np.array(range(0,l//2))
    adaptation_indices[a] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]
    #这一步是内循环里面的支持集训练
    for step in range(adaptation_steps):
        train_error = loss(learner(adaptation_data), adaptation_labels)
        learner.adapt(train_error)
    #这一步是内循环里面的查询集评估刚才支持集训练的模型
    predictions = learner(evaluation_data)
    valid_error = loss(predictions, evaluation_labels)
    valid_accuracy = accuracy(predictions, evaluation_labels)
    return valid_error, valid_accuracy


def main(
   args
):
    device = torch.device('cuda:4')
    #将目标任务去除创建任务集
    tasknames1=["电缆"]
    for taskname in tasknames1:
        task_names = os.listdir(args.path)
        task_names.remove(taskname)
        tasksets = MyData(args.path, task_names, mode='train', n_way=args.n_way, k_shot=args.k_spt, k_query=args.k_qry,
                          batchsz=args.task_num, resize=args.imgsz)
        dataloader = DataLoader(tasksets, args.meta_bsz, shuffle=False, num_workers=0, pin_memory=True)
        #加载模型
        model = Ours()
        features1 = model.S_net
        features2 = model.C_net1
        features3 = model.ca
        features4 = model.C_net2
        features5 = model.C_net3
        features6 = model.C_net4
        head = model.classifier
        features1.to(device)
        features2.to(device)
        features3.to(device)
        features4.to(device)
        features5.to(device)
        features6.to(device)
        head.to(device)
        # 设置优化器和损失函数
        head = l2l.algorithms.MAML(head, lr=args.meta_lr)
        all_parameters = list(features1.parameters())+list(features2.parameters()) +list(features3.parameters()) \
                         +list(features4.parameters()) +list(features5.parameters()) + list(features6.parameters()) +\
                         list(head.parameters())
        optimizer = torch.optim.Adam(all_parameters, lr=args.meta_lr)
        loss = nn.CrossEntropyLoss(reduction='mean')
        #maml训练
        for iteration in range(args.epoch):
            meta_train_error = 0.0
            meta_train_accuracy = 0.0
            for iter, (data_x, data_y, data_x_0) in enumerate(dataloader,0):
                #内循环
                all_parameters[-1].data.zero_()
                all_parameters[-2].data.zero_()
                evaluation_errors = torch.zeros(1).to(device)
                for i in range(0,data_x.size(0)):
                    learner = head.clone()
                    evaluation_error, evaluation_accuracy = fast_adapt(
                                                                       (data_x[i], data_y[i], data_x_0[i]),
                                                                       learner,
                                                                       (features1,features2,features3,features4,features5,features6),
                                                                        loss,
                                                                        args.adaptation_steps,
                                                                        device
                                                                       )
                    evaluation_errors=evaluation_errors+evaluation_error
                    meta_train_error += evaluation_error.item()
                    meta_train_accuracy += evaluation_accuracy.item()
                #外循环
                optimizer.zero_grad()
                evaluation_errors.backward()
                for p in all_parameters:
                    p.grad.data.mul_(1.0 / args.meta_bsz)
                optimizer.step()
            #结果输出
            print('Iteration', iteration+1)
            print('Meta Train Error', meta_train_error / args.task_num)
            print('Meta Train Accuracy', meta_train_accuracy / args.task_num)
            #每50次保存一次权重
            if((iteration+1)%50==0):
                torch.save(model,"./mypkl/"+taskname+"/test_base"+str(iteration+1)+".pkl")
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    #数据设置
    argparser.add_argument('--path', type=str, help="数据集地址",default="./dataset/")
    argparser.add_argument('--k_spt', type=int, help='支持集比例', default=7)
    argparser.add_argument('--k_qry', type=int, help='查询集比例', default=3)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=256)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)

    # maml相关参数
    argparser.add_argument('--epoch', type=int, help='epoch number', default=1000)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.001)
    argparser.add_argument('--fast_lr', type=float, help='task-level inner update learning rate', default=1e-5)
    argparser.add_argument('--meta_bsz', type=int, help='task_batch', default=1)
    argparser.add_argument('--adaptation_steps', type=int, help='inner loop iter', default=1)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=29)
    argparser.add_argument('--n_way', type=int, help='n way', default=2)

    args = argparser.parse_args()
    main(args)
