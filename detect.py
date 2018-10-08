import torch
import numpy as np
import constant
from nets import PNet, RNet, ONet
import os.path as opath
from torchvision import transforms
import util
from PIL import Image, ImageDraw, ImageFont


class FaceDetector:

    def __init__(self, pnet_param_path=constant.PNet_param_path, rnet_param_path=constant.RNet_param_path,
                 onet_param_path=constant.ONet_param_path):
        self.p_net = PNet()
        self.r_net = RNet()
        self.o_net = ONet()

        if opath.exists(pnet_param_path):
            self.p_net.load_state_dict(torch.load(pnet_param_path))

        if opath.exists(rnet_param_path):
            self.r_net.load_state_dict(torch.load(rnet_param_path))

        if opath.exists(onet_param_path):
            self.o_net.load_state_dict(torch.load(onet_param_path))

        self.p_net.eval()
        self.r_net.eval()
        self.o_net.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __onet_detect(self, image, rnet_boxes):
        _img_dataset = []
        _rnet_boxes = util.convert_to_square(rnet_boxes)
        for _box in _rnet_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            img = image.crop((_x1, _y1, _x2, _y2))
            img = img.resize((48, 48), Image.ANTIALIAS)
            img_data = self.transform(img)
            _img_dataset.append(img_data)

        img_dataset = torch.stack(_img_dataset)
        _cls, _offset = self.o_net(img_dataset)

        cls = _cls.detach().numpy()
        offset = _offset.detach().numpy()

        boxes = []
        idxs, _ = np.where(cls > 0.97)
        for idx in idxs:
            _box = _rnet_boxes[idx]
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            ow = _x2 - _x1
            oh = _y2 - _y1

            x1 = _x1 + ow * offset[idx][0]
            y1 = _y1 + oh * offset[idx][1]
            x2 = _x2 + ow * offset[idx][2]
            y2 = _y2 + oh * offset[idx][3]

            boxes.append([x1, y1, x2, y2, cls[idx][0]])
        # O网络最后使用(交集/最小值)的方法做IOU运算
        return util.NMS(np.array(boxes), isMin=True, thresh=0.7)

    def __rnet_detect(self, image, pnet_boxes):
        pnet_boxes_ = util.convert_to_square(pnet_boxes)  # 转正方形
        img_datas = []
        for box in pnet_boxes_:
            x1_ = int(box[0])
            y1_ = int(box[1])
            x2_ = int(box[2])
            y2_ = int(box[3])

            img = image.crop((x1_, y1_, x2_, y2_))
            img = img.resize((24, 24), Image.ANTIALIAS)
            img_data = self.transform(img)
            img_datas.append(img_data)

        img_datas = torch.stack(img_datas)

        cond, offset = self.r_net(img_datas)  # cond的形式为(batch,1), offset的形式为(batch,4)

        cond = cond.detach().numpy()
        offset = offset.detach().numpy()

        indexs, _ = np.where(cond > 0.7)
        boxes = []

        for index in indexs:
            box = pnet_boxes_[index]
            _x1 = int(box[0])
            _y1 = int(box[1])
            _x2 = int(box[2])
            _y2 = int(box[3])

            ow = _x2 - _x1
            oh = _y2 - _y1

            x1 = offset[index][0] * ow + _x1
            y1 = offset[index][1] * oh + _y1
            x2 = offset[index][2] * ow + _x2
            y2 = offset[index][3] * oh + _y2

            boxes.append([x1, y1, x2, y2, cond[index][0]])

        return util.NMS(np.array(boxes), thresh=0.5)

    def __pnet_detect(self, image):

        boxes = []

        img = image
        w, h = img.size
        min_side_len = min(w, h)  # 最小边长

        scale = 1  # 缩放比例

        # 通过循环生成图像金字塔
        while min_side_len > 12:  # P网络建议框长度为12
            img_data = self.transform(img)  # 向量化
            img_data.unsqueeze_(0)  # 升维，由 CHW 转为 1CHW,相当于加了个批次

            cond, offset = self.p_net(img_data)  # 1CHW
            cond_ = cond[0][0]
            offset_ = offset[0]

            cond_mask = cond_ > 0.6  # 置信度大于0.6
            indexs = torch.nonzero(cond_mask)  # 找出置信度大于0.6的下标

            for index in indexs:
                orignal_box = self.__restore_box(index, cond_[index[0], index[1]], offset_, scale)
                boxes.append(orignal_box)

            rate = 0.9
            scale *= rate
            _w = int(w * scale)
            _h = int(h * scale)

            img = img.resize([_w, _h])
            min_side_len = min(_w, _h)

        return util.NMS(np.array(boxes), 0.5)  # nms操作

    # 将回归量还原到原图上
    # PNet的全卷机模型，所有卷积和池化的步长相乘为2，所以这里stride=2,而PNet的默认边长是12，所以这里side_len=12
    def __restore_box(self, start_index_, cls, offset_, scale, stride=2, side_len=12):

        start_index = start_index_.numpy()
        offset = offset_.detach().numpy()
        # 在原图上的建议框
        _x1 = (start_index[1] * stride) / scale  # w
        _y1 = (start_index[0] * stride) / scale  # h
        _x2 = (start_index[1] * stride + side_len) / scale  # w
        _y2 = (start_index[0] * stride + side_len) / scale  # h

        ow = _x2 - _x1
        oh = _y2 - _y1

        _offset = offset[:, start_index[0], start_index[1]]
        x1 = round(_offset[0] * ow + _x1)
        y1 = round(_offset[1] * oh + _y1)
        x2 = round(_offset[2] * ow + _x2)
        y2 = round(_offset[3] * oh + _y2)

        return [x1, y1, x2, y2, cls.detach().numpy()]

    def detect(self, image):
        # P网络侦测
        pnet_boxes = self.__pnet_detect(image)
        if pnet_boxes.shape[0] == 0:
            return np.array([])

        # return pnet_boxes
        # R网络侦测
        rnet_boxes = self.__rnet_detect(image, pnet_boxes)
        if rnet_boxes.shape[0] == 0:
            return np.array([])

        # O网络侦测
        onet_boxes = self.__onet_detect(image, rnet_boxes)
        if onet_boxes.shape[0] == 0:
            return np.array([])

        return onet_boxes


if __name__ == '__main__':
    image_file = r"C:\Users\Administrator\Desktop\图片\777.jpg"
    # image_file = r"D:\AI\celebA\CelebA\Img\img_celeba\img_celeba\000020.jpg"
    facedetector = FaceDetector()
    with Image.open(image_file) as img:
        imDraw = ImageDraw.Draw(img)
        boxes = facedetector.detect(img)
        for box in boxes:
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])

            cond = box[4]

            imDraw.rectangle((x1, y1, x2, y2), outline="red")
            imDraw.text((x1, y1), str(cond))

        img.show()
