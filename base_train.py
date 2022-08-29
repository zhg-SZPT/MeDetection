from util.MyDataSub import MyDataSub
import numpy as np
import torch
from torch import nn
from    torch.nn import functional as F
import util.MyUtils as ut
from util.MyBaseData  import MyData
from    torch.utils.data import DataLoader
from util.ours import Ours
import math
import  argparse
from sklearn.metrics import recall_score,f1_score,precision_score,roc_auc_score
def main(args):
    device = torch.device('cuda:5')
    #结果存储以什么开头
    savename="maml-guide"
    # tasknames=["药片","金属栅栏", "晶体管","电缆","金属螺母", "螺丝钉","牙刷","胶囊"]
    tasknames=["电缆"]
    pkl="test420.pkl"
    for taskname in tasknames:
        #数据
        data_test = MyData(args.path, [taskname,taskname], mode='test', n_way=args.n_way, k_shot=args.k_spt_dst,
                           k_query=args.k_qry_dst, batchsz=args.task_num, resize=args.imgsz)
        x_spt, y_spt, x_qry, y_qry,x_spt_0,imgs_name = data_test.__getitem__(0)
        x_spt, y_spt, x_qry, y_qry,x_spt_0 = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                     x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device),x_spt_0.squeeze(0).to(device)
        train_data = MyDataSub(x_spt, y_spt,imgs_name,flag="train")
        train_dataloader = DataLoader(train_data,32, shuffle=True, num_workers=0, pin_memory=False)
        test_data  = MyDataSub(x_qry, y_qry,imgs_name,flag="test")
        test_dataloader = DataLoader(test_data,32, shuffle=True, num_workers=0, pin_memory=False)
        #模型
        model1 = torch.load("./mypkl/" + taskname + "/" + pkl, map_location='cpu')
        dict=model1.state_dict()
        # del dict["classifier.0.bias"]
        # del dict["classifier.0.weight"]
        model=Ours()
        classifier = nn.Sequential(
            nn.Linear(672, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 2),
        )
        # model.classifier=classifier
        # for k,v in dict.items():
        #     dict1[k]=v
        # 首先获取全连接层参数的地址
        fc_params_id = list(map(id, model.classifier.parameters()))  # 返回的是parameters的 内存地址
        # 然后使用 filter 过滤不属于全连接层的参数，也就是保留卷积层的参数
        base_params = filter(lambda p: id(p) not in fc_params_id, model.parameters())
        # 设置优化器的分组学习率，传入一个 list，包含 2 个元素，每个元素是字典，对应 2 个参数组
        net = model.to(device)
        #优化器与损失
        criterion = nn.CrossEntropyLoss()
        # optimizer = torch.optim.Adam(net.parameters(), lr=args.update_lr)
        optimizer = torch.optim.Adam(
            [{'params': base_params, 'lr': 1e-7}, {'params': model.classifier.parameters(), 'lr': 1e-5}])
        #评估标准集合
        correts=[]
        recalls=[]
        precisions=[]
        f1s=[]
        aucs=[]

        #模型训练
        for epoch in range(0, args.update_step_test):
            train_correct = 0
            r=[200,224, 256, 300,320]
            for ratio in r:
                for i, data in enumerate(train_dataloader, 0):
                    t_img0, label = data
                    t_img1= ut.getBatchZeroSampler(x_spt_0,t_img0.shape[0])
                    _,_,h,w=t_img0.size()
                    h_new = ratio
                    w_new = ratio
                    img0 = F.interpolate(t_img0, [h_new, w_new], mode='nearest')
                    img1 = F.interpolate(t_img1, [h_new, w_new], mode='nearest')
                    optimizer.zero_grad()
                    result = net.forward(img0, img1)
                    label = label.squeeze()
                    loss_contrastive = criterion(result, label)
                    loss_contrastive.backward()
                    optimizer.step()
                    pred = torch.argmax(result.data, 1)
                    train_correct += pred.eq(label.data).cpu().sum()
            print("Epoch number: {} , Current loss: {:.4f}".format(epoch + 1, loss_contrastive.item()))
            train_acc = float(train_correct) /(x_spt.shape[0]*len(r))
            print('train_acc:', train_acc)

            #模型测试
            with torch.no_grad():
                test_correct = 0
                ratios=[200,224, 256, 300,320]
                y_qry_true = torch.zeros(0)
                y_qry_pre = torch.zeros(0)
                for ratio in ratios:
                    for i, data in enumerate(test_dataloader, 0):
                        t_img0, label,_ = data
                        t_img1 = ut.getBatchZeroSampler(x_spt_0, t_img0.shape[0])
                        _, _, h, w = t_img0.size()
                        h_new = math.ceil(ratio)
                        w_new = math.ceil(ratio)
                        img0 = F.interpolate(t_img0, [h_new, w_new], mode='nearest')
                        img1 = F.interpolate(t_img1, [h_new, w_new], mode='nearest')
                        result = net.forward(img0, img1)
                        label = label.squeeze()
                        pred = torch.argmax(result.data, 1)
                        y_qry_true = torch.hstack((y_qry_true, label.data.cpu()))
                        y_qry_pre = torch.hstack((y_qry_pre, pred.cpu()))
                        test_correct += pred.eq(label.data).cpu().sum()
            r=len(ratios)
            test_acc = float(test_correct) / (x_qry.shape[0]*r)
            correts.append(test_acc)
            recall = recall_score(np.array(y_qry_true.cpu()), np.array(y_qry_pre.cpu()))
            recalls.append(recall)
            precision = precision_score(np.array(y_qry_true.cpu()), np.array(y_qry_pre.cpu()))
            precisions.append(precision)
            f1 = f1_score(np.array(y_qry_true.cpu()), np.array(y_qry_pre.cpu()))
            f1s.append(f1)
            auc = roc_auc_score(np.array(y_qry_true.cpu()), np.array(y_qry_pre.cpu()))
            aucs.append(auc)
            print('test_acc,recall,precision,f1,auc:', test_acc, recall, precision, f1,auc)

        # 权重存储
        torch.save(model,"./mypkl/"+taskname+"/"+savename+".pkl")
        # 结果存储
        co=np.array(correts)
        np.save("./mynpy/"+taskname+"/"+savename+"-c.npy", co)
        co=np.array(recalls)
        np.save("./mynpy/"+taskname+"/"+savename+"-r.npy", co)
        co=np.array(precisions)
        np.save("./mynpy/"+taskname+"/"+savename+"-p.npy", co)
        co=np.array(f1s)
        np.save("./mynpy/"+taskname+"/"+savename+"-f.npy", co)
        co=np.array(aucs)
        np.save("./mynpy/"+taskname+"/"+savename+"-a.npy", co)
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    # 数据设置
    argparser.add_argument('--path', type=str, default="/zzh/test/diff1/dataset/")
    argparser.add_argument('--k_spt_dst', type=int, help='训练集', default=7)
    argparser.add_argument('--k_qry_dst', type=int, help='测试集', default=3)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=256)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)

    # 目标任务相关参数
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=2000)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=1e-5)
    argparser.add_argument('--n_way', type=int, help='n way', default=2)

    # 其他参数
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=29)

    args = argparser.parse_args()
    main(args)