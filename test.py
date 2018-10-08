import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class PNet(nn.Module):

    def __init__(self):
        super(PNet, self).__init__()

        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=3, stride=1),  # conv1
            nn.PReLU(),  # PReLU1
            nn.MaxPool2d(kernel_size=2, stride=2),  # pool1
            nn.Conv2d(10, 16, kernel_size=3, stride=1),  # conv2
            nn.PReLU(),  # PReLU2
            nn.Conv2d(16, 32, kernel_size=3, stride=1),  # conv3
            nn.PReLU()  # PReLU3
        )

        self.conv4_1 = nn.Conv2d(32, 1, kernel_size=1, stride=1)
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.pre_layer(x)
        cond = torch.sigmoid(self.conv4_1(x))
        offset = self.conv4_2(x)
        return cond, offset


import numpy as np

# aa = torch.Tensor(np.arange(1, 1 * 3 * 12 * 12 + 1).reshape(1, 3, 12, 12))
# p = PNet()
#
# cls, offset = p(aa)
#
# cls_ = cls[0][0]
# offset_ = offset[0]
# # print(cls_)
#
# # print("======================")
#
# indexs = torch.nonzero(torch.gt(cls_, 0.6))
#
# print(offset_)
# print(offset_[:, 0, 0])

# for index in indexs:
#     print(indexs)
#     print(index)
#     print(cls_[index[0], index[1]])
#     print(cls_[index[0]][index[1]])


# print("======================")
# aa = torch.Tensor([[1, 2], [3, 4], [5, 6]])
# bb = torch.Tensor([[3]])
# print(aa < 4)
# print(torch.nonzero(bb < 4))
#
# print("-----------------------")
# print(torch.nonzero(torch.Tensor([[[5]]])))
# aa = torch.Tensor([[1], [2], [3], [4], [5]])
# cc = np.where(aa <= 3)
# index, b = cc
#
# print(cc)
#
# dd = torch.nonzero(aa <= 3)
# print(dd)

# x = np.arange(16).reshape(-1, 4)
# print(x)
# print(np.where(x > 5))

# a = torch.Tensor([[1, 2], [3, 4], [5, 6]])
# b = torch.Tensor([[7, 8], [9, 10], [11, 12]])
#
# aa = torch.stack([a, b], dim=0)
# print(aa)
# print(torch.stack([a, b], dim=1))
# print(torch.stack([a, b], dim=2))

# a = np.array([[1, 2], [4, 9], [5, 6]])
# print(np.max(a, axis=0))
# print(np.max(a, axis=1))
#
# print(np.sum(a, axis=0))
# print(np.sum(a, axis=1))

# x = torch.Tensor([[1, 2, 5], [3, 4, 10], [5, 6, 20]])
# print(x)
#
# indices = torch.LongTensor([0, 2])
# y = torch.index_select(x, 0, indices)
# print(y)
#
# z = torch.index_select(x, 1, indices)
# print(z)


# aa = torch.Tensor([0, 1, 0, 2])
# bb = torch.histc(aa, bins=2, max=1, min=0)
# print(torch.histc(aa))
# print(torch.histc(aa, bins=4, max=1, min=0))


# print(type(np.random.randint(1, 3)))
# print(type(np.random.uniform(1, 3)))


# aa = torch.Tensor(np.arange(1, 1 * 4 * 2 * 3 + 1).reshape([1, 4, 2, 3]))  # NCHW
# aa = aa.transpose(1, 3).transpose(2, 1)
# print(aa)
# print(aa.shape)
# aa = aa.view(-1, 4)
# print(aa)

# print(cc)

# print(bb)
#
# bb = bb.view((-1, 4))
# print(bb)


# import constant
# from PIL import ImageDraw, Image
#
# str = constant.Anno_path
#
# for index, strs in enumerate(open(str)):
#     if index < 2:
#         continue
#     print(index - 1, strs)


# import torch
# from PIL import Image
#
# str = r"C:\Users\Administrator\Desktop\777.jpg"
# with Image.open(str) as img:
#     im = img.resize((48, 48))
#     im.save(r"C:\Users\Administrator\Desktop\777_1.jpg")

# aa = torch.Tensor([[1, 8], [2, 6], [3, 2]])
# print(torch.nonzero(aa <= 3))
#
# bb = aa.numpy()
# cc = np.where(bb <= 3)
# print(cc)


aa = torch.Tensor(np.arange(1, 25).reshape([1, 3, 2, 4]))  # NCHW
print(aa[0][1])  # c = 1
print(aa[0][1][1][2])  # h = 1, w = 2

cc = aa.permute(0, 2, 3, 1)  # NHWC

print(cc[0][1][2][1])
