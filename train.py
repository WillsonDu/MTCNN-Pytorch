import torch
import torch.nn as nn
from nets import PNet, RNet, ONet
from torch.utils.data import DataLoader
import os
from facedataset import FaceDataSet
import constant
import traceback


class Trainer:

    def __init__(self, net, dataset_path, net_params_path):
        self.net = net
        self.dataset_path = dataset_path
        self.net_params_path = net_params_path

        self.opt = torch.optim.Adam(net.parameters())  # 优化器
        self.cond_loss_fn = nn.BCELoss()  # 两个数的交叉熵
        self.offset_loss_fn = nn.MSELoss()  # 均方差

        if os.path.exists((self.net_params_path)):  # 加载参数
            net.load_state_dict(torch.load(self.net_params_path))

    def train(self):
        facedateset = FaceDataSet(self.dataset_path)
        dataloader = DataLoader(facedateset, batch_size=100, shuffle=True, num_workers=2)
        # for _ in range(10000):
        try:
            for index, (img_data, cond, offset) in enumerate(dataloader):
                print("正常数据！")
                cond_out, offset_out = self.net(img_data)  # 训练
                cond_out = cond_out.view([-1, 1])
                offset_out = offset_out.view([-1, 4])

                # 计算cond置信度的损失
                cond_mask = cond < 2  # 部分样本的置信度为2，不参与其对置信度的计算
                cond_ = cond[cond_mask]
                cond_out = cond_out[cond_mask]
                cond_loss = self.cond_loss_fn(cond_out, cond_)

                # 计算偏移offset的损失
                offset_mask = cond > 0  # 负样本的置信度为0，不参与其对偏移offset的计算
                offset_ = offset[offset_mask[:, 0]]
                offset_out = offset_out[offset_mask[:, 0]]
                offset_loss = self.offset_loss_fn(offset_out, offset_)

                # 总损失
                loss = cond_loss + offset_loss

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                print("index:", index, ", loss:", loss, ", cond_loss:", cond_loss, ", offset_loss:", offset_loss)

                # 保存变量数据
                torch.save(self.net.state_dict(), self.net_params_path)  # 保存网络变量
            print("整批数据训练完毕！！！")
        except:
            # print("数据中含有None，跳过这批数据！")
            traceback.print_exc()
            # continue


if __name__ == '__main__':
    # # 训练P网络
    # net = PNet()
    # trainer = Trainer(net, constant.Data_save_path + r"\12", constant.PNet_param_path)
    # trainer.train()

    # # 训练R网络
    # net = RNet()
    # trainer = Trainer(net, constant.Data_save_path + r"\24", constant.RNet_param_path)
    # trainer.train()

    # 训练O网络
    net = ONet()
    trainer = Trainer(net, constant.Data_save_path + r"\48", constant.ONet_param_path)
    trainer.train()
