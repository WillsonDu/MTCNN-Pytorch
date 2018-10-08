import torch
from torch.utils.data import Dataset
import os
import os.path as opath
from torchvision import transforms
from PIL import Image
import numpy as np


class FaceDataSet(Dataset):

    def __init__(self, path):
        super(FaceDataSet, self).__init__()

        self.tf = transforms.Compose([
            transforms.ToTensor()
        ])

        self.img_path = opath.join(path, "images")
        self.dataset = []

        posi_path = opath.join(path, "positive.txt")
        nega_path = opath.join(path, "negative.txt")
        part_path = opath.join(path, "part.txt")

        if opath.exists(posi_path):
            self.dataset.extend(open(posi_path).readlines())
        if opath.exists(nega_path):
            self.dataset.extend(open(nega_path).readlines())
        if opath.exists(part_path):
            self.dataset.extend(open(part_path).readlines())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        strs = item.split()
        cond = torch.Tensor([float(strs[1])])  # 置信度
        # 偏移量
        offset = torch.Tensor([float(strs[2]), float(strs[3]), float(strs[4]), float(strs[5])])

        # img
        img_path = os.path.join(self.img_path, strs[0])  # 文件名称

        if not opath.exists(img_path):  # 如果文件不存在，返回None. DataLoader处会抛异常，在那里特殊处理就是了
            return None

        img_data = np.array(Image.open(img_path))
        img_data = self.tf(img_data)  # 归一化, 并转换成Tensor
        img_data -= 0.5  # 去均值

        return img_data, cond, offset
