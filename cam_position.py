import cv2
from util.MyBaseData import MyData
from PIL import Image
from torch.autograd import Variable
from util.MyDataSub import MyDataSub
import numpy as np
import torch
from    torch.nn import functional as F
import util.MyUtils as ut
from    torch.utils.data import  DataLoader
import  argparse
from   util.cam import Cam

class GradCam:
    def __init__(self, model, result, features):
        self.model = model
        self.result = result
        self.features = features

    def get_cam_weights(self, target, grads):
        grads_power_2 = grads ** 2
        grads_power_3 = grads_power_2 * grads
        sum_activations = np.sum(target, axis=(2, 3))
        eps = 0.000001
        aij = grads_power_2 / (2 * grads_power_2 +
                               sum_activations[:, :, None, None] * grads_power_3 + eps)
        aij = np.where(grads != 0, aij, 0)

        weights = np.maximum(grads, 0) * aij
        weights = np.sum(weights, axis=(2, 3))
        return weights

    def __call__(self, index):
        if index is None:
            index = np.argmax(self.result.cpu().data.numpy(), 1)

        one_hot = np.zeros(self.result.size(), dtype=np.float32)
        for i in range(self.result.size(0)):
            one_hot[i][index[i]] = 1
        # one_hot = torch.Tensor(torch.from_numpy(one_hot))
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        global device
        one_hot = torch.sum(one_hot.to(device) * self.result)
        self.model.zero_grad()
        one_hot.backward()
        grads_val = self.model.gradients[-1].cpu().data.numpy()     # 32 x 512 x 7 x 7
        target = self.features.cpu().data.numpy()          # 32 x 512 x 7 x 7
        # weights = np.mean(grads_val, axis=(2, 3))           # 32 x 512
        weights = self.get_cam_weights(target, grads_val)
        cam = np.zeros((target.shape[0], 224, 224), dtype=np.float32)
        for i in range(target.shape[0]):
            cam_para = np.zeros(target.shape[2:], dtype=np.float32)
            for j, w in enumerate(weights[i]):
                cam_para += w * target[i, j, :, :]

            cam_para = np.maximum(cam_para, 0)
            cam_para = cv2.resize(cam_para, (224, 224))
            cam_para = cam_para -np.min(cam_para)
            cam_para = cam_para / np.max(cam_para)
            cam[i] = cam_para
        return cam
def main(args):
    global device
    device = torch.device('cuda:3')
    tasknames = {"地毯":"tiles","瓷砖":"carpet","瓶盖":"cap","人行道":"sidewalk", "面料1":"fabric1", "面料4":"fabric2", "面料6":"fabric3", "面料10":"fabric4"}
    tasknames = {"瓶子": "bottle", "晶体管": "transistor", "木材": "wood", "榛子": "hazelnut", "电缆": "cable", "药片": "pill",
                 "螺丝钉": "screw", "金属栅栏": "grid", "金属螺母": "metal nut"
                 }
    pkls=["random","maml"]
    for taskname,taskname1 in tasknames.items():
        #加载任务集
        data_test = MyData(args.path, [taskname, taskname], mode='test', n_way=args.n_way, k_shot=args.k_spt_dst,
                               k_query=args.k_qry_dst, batchsz=args.task_num, resize=args.imgsz)
        x_spt, y_spt, x_qry, y_qry, x_spt_0, imgs_path = data_test.__getitem__(0)
        x_spt, y_spt, x_qry, y_qry, x_spt_0 = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                              x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device), x_spt_0.squeeze(
            0).to(device)
        test_data = MyDataSub(x_qry, y_qry, imgs_path, flag="test")
        test_dataloader = DataLoader(test_data, 60, shuffle=True, num_workers=0, pin_memory=False)
        for pkl in pkls:
            print(taskname+"-"+pkl)
            #加载训练权重
            model = Cam()
            model1 = torch.load("./mypkl/" + taskname+ "/" + pkl+".pkl", map_location='cpu')
            model.load_state_dict(model1.state_dict())
            net = model.to(device)
            for epoch in range(0, 1):
                test_correct = 0
                ratios=[224]
                y_qry_true = torch.zeros(0)
                y_qry_pre = torch.zeros(0)
                for ratio in ratios:
                    for i, data in enumerate(test_dataloader, 0):
                        t_img0, label,imgname = data
                        t_img1 = ut.getBatchZeroSampler(x_spt_0, t_img0.shape[0])
                        _, _, h, w = t_img0.size()
                        h_new = ratio
                        w_new = ratio
                        img0 = F.interpolate(t_img0, [h_new, w_new], mode='nearest')
                        img1 = F.interpolate(t_img1, [h_new, w_new], mode='nearest')
                        img0, img1, label = img0.to(device), img1.to(device), label.to(device)
                        result, feature = net.forward(img0, img1)
                        label = label.squeeze()
                        pred = torch.argmax(result.data, 1)
                        y_qry_true = torch.hstack((y_qry_true, label.data.cpu()))
                        y_qry_pre = torch.hstack((y_qry_pre, pred.cpu()))
                        test_correct += pred.eq(label.data).cpu().sum()
                        #Cam算法
                        gradcam = GradCam(net, result, feature)
                        target_index = None
                        mask = gradcam(target_index)
                        if(ratio==224):
                            for j in range(len(label)):
                                if pred[j] == 1:
                                    index = imgname[j]
                                    name=index.split("/")[-1]
                                    heatmap = cv2.applyColorMap(np.uint8(255 * mask[j]), cv2.COLORMAP_JET)
                                    heatmap = np.float32(heatmap) / 255
                                    cam = heatmap
                                    cam = cam / np.max(cam)
                                    cam = cv2.resize(cam, (224, 224))

                                    path="./output/"+taskname1+"/"+pkl+"/cam/"+name
                                    cv2.imwrite(path,np.uint8(255 * cam))

                                    img = Image.open(index).convert('RGB')
                                    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
                                    img = np.float32(cv2.resize(img, (224, 224)))

                                    cv2.imwrite("./output/"+taskname1+"/"+pkl+"/real/"+name, img)

                                    result = heatmap + np.float32(img) / 255
                                    result = result / np.max(result)
                                    result = cv2.resize(result, (224, 224))
                                    result = np.uint8(255 * result)
                                    mask_para = np.uint8(255 * mask[j])
                                    for h in range(224):
                                        for w in range(224):
                                            for c in range(3):
                                                if mask_para[h, w] < 128:
                                                    result[h, w, c] = img[h, w, c]

                                    cv2.imwrite("./output/"+taskname1+"/"+pkl+"/res/"+name, result)
                                    print(name)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    # 数据设置
    argparser.add_argument('--path', type=str, default="/zzh/test/diff1/dataset/")
    argparser.add_argument('--k_spt_dst', type=int, help='训练集', default=7)
    argparser.add_argument('--k_qry_dst', type=int, help='测试集', default=3)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=256)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)

    # 目标任务相关参数
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=1)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=1e-5)
    argparser.add_argument('--n_way', type=int, help='n way', default=2)

    # 其他参数
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=29)

    args = argparser.parse_args()
    main(args)