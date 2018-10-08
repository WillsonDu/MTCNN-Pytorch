import torch as tc
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# 普通卷积
def conv_bn(in_channel, out_channel, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channel),
        nn.PReLU()
    )


class PNet(nn.Module):  # (batch,3,12,12)

    def __init__(self):
        super(PNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=3, padding=0),  # (batch,6,10,10)
            nn.BatchNorm2d(6),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # (batch,6,5,5)

            nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=0),  # (batch,16,3,3)
            nn.BatchNorm2d(16),
            nn.PReLU(),

            nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=0),  # (batch,64,1,1)
            # nn.BatchNorm2d(64),
            nn.PReLU(),
        )

        self.conv_1 = nn.Conv2d(64, 1, kernel_size=1, stride=1)  # (batch,1,1,1)
        self.conv_4 = nn.Conv2d(64, 4, kernel_size=1, stride=1)  # (batch,4,1,1)

    def forward(self, x):
        x = self.conv1(x)
        cond = tc.sigmoid(self.conv_1(x))
        offset = self.conv_4(x)
        return cond, offset


class RNet(nn.Module):  # (batch,3,24,24)

    def __init__(self):
        super(RNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 28, kernel_size=3, padding=1),  # (batch,28,24,24)
            nn.BatchNorm2d(28),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (batch,28,11,11)

            nn.Conv2d(28, 48, kernel_size=2, padding=1, stride=2),  # (batch,48,6,6)
            nn.BatchNorm2d(48),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1),  # (batch,48,4,4)

            nn.Conv2d(48, 64, kernel_size=2, stride=1),  # (batch,64,3,3)
            nn.BatchNorm2d(64),
            nn.PReLU(),
        )
        # 全连接
        self.linear1 = nn.Sequential(
            nn.Linear(64 * 3 * 3, 128),
            nn.PReLU()
        )

        self.cond = nn.Linear(128, 1)  # (batch,1)
        self.offset = nn.Linear(128, 4)  # (batch,4)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.shape[0], -1)
        x = self.linear1(x)

        cond =tc.sigmoid(self.cond(x))
        offset = self.offset(x)
        return cond, offset


class ONet(nn.Module):  # (batch,3,48,48)

    def __init__(self):
        super(ONet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1),  # (batch,32,46,46)
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # (batch,32,23,23)

            nn.Conv2d(32, 64, kernel_size=3, stride=1),  # (batch,64,21,21)
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (batch,64,10,10)

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),  # (batch,64,8,8)
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (batch,64,4,4)

            nn.Conv2d(64, 128, kernel_size=2, stride=1),  # (batch,128,3,3)
            nn.BatchNorm2d(128),
            nn.PReLU()
        )

        # 全连接
        self.lienar1 = nn.Sequential(
            nn.Linear(128 * 3 * 3, 256),
            nn.PReLU()
        )

        self.cond = nn.Linear(256, 1)
        self.offset = nn.Linear(256, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.shape[0], -1)
        x = self.lienar1(x)

        cond = tc.sigmoid(self.cond(x))
        offset = self.offset(x)

        return cond, offset


if __name__ == '__main__':
    pnet = PNet()
    for _ in range(100):
        x = tc.Tensor(np.arange(1, 3 * 12 * 12 + 1).reshape([1, 3, 12, 12]))
        cond, offset = pnet(x)
        print(cond.shape, offset.shape)
