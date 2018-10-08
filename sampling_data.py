import torch as tc
import os
import os.path as opath
import numpy as np
import util
import traceback
import constant
from PIL import Image

anno_path = constant.Anno_path
original_img_path = constant.Original_img_path
save_path = constant.Data_save_path


class Sampling:

    def mkdir(self, face_size):
        rootpath = opath.join(save_path, str(face_size))
        if opath.exists(rootpath) == False:
            os.mkdir(rootpath)

        save_img_path = opath.join(rootpath, "images")
        if opath.exists(save_img_path) == False:
            os.mkdir(save_img_path)

        return rootpath, save_img_path

    def sample(self, face_size):

        # 创建文件夹(样本图片保存路径)
        rootpath, save_img_path = self.mkdir(face_size)

        positive_count = 0
        negative_count = 0
        part_count = 0

        try:
            # 描述样本存储路径
            positive_file = open(opath.join(rootpath, "positive.txt"), "w")
            negative_file = open(opath.join(rootpath, "negative.txt"), "w")
            part_file = open(opath.join(rootpath, "part.txt"), "w")

            for index, line in enumerate(open(anno_path)):

                if index <= 1:  # 文本第三行开始才是有效信息
                    continue

                strs = line.split()  # 拆分字符串
                filename = strs[0]  # 文件名

                print("当前扫描文件: ", filename, ". positive_count: ", positive_count, " ,negative_count: ", negative_count,
                      " ,part_count: ", part_count)

                img_file = opath.join(original_img_path, filename)
                if not opath.exists(img_file):
                    continue

                # x,y坐标
                x1 = float(strs[1])
                y1 = float(strs[2])
                w = float(strs[3])
                h = float(strs[4])
                x2 = float(x1 + w)
                y2 = float(y1 + h)

                # 最大边长
                max_side = np.maximum(w, h)

                # 人脸中心点
                center_x = x1 + w / 2
                center_y = y1 + h / 2

                try:

                    with Image.open(img_file) as img:
                        img_width, img_height = img.size  # 图片宽高

                        # 创建正样本、负样本、部分样本(此处负样本几率较小)
                        for _ in range(5):
                            # 边长浮动
                            _side = np.random.uniform(0.8, 1.2) * max_side

                            # 中心点浮动
                            offset_x = np.random.uniform(-0.2, 0.2) * w / 2
                            offset_y = np.random.uniform(-0.2, 0.2) * h / 2
                            _center_x = center_x + offset_x
                            _center_y = center_y + offset_y

                            # 根据浮动计算浮动后的x和y
                            _x1 = np.maximum(0, _center_x - _side * 0.5)
                            _y1 = np.maximum(0, _center_y - _side * 0.5)
                            _x2 = _x1 + _side
                            _y2 = _y1 + _side

                            # 计算偏移值
                            offset_x1 = (x1 - _x1) / _side
                            offset_y1 = (y1 - _y1) / _side
                            offset_x2 = (x2 - _x2) / _side
                            offset_y2 = (y2 - _y2) / _side

                            box = np.array([_x1, _y1, _x2, _y2])
                            boxes = np.array([[x1, y1, x2, y2]])

                            crop_img = img.crop(box)  # 从原图中裁剪出目标图像
                            crop_img = crop_img.resize((face_size, face_size))
                            ious = util.IOU(box, boxes, isMin=False)
                            iou = ious[0]

                            if iou < 0.3:  # 负样本(在这里几率比较小)
                                negative_count += 1.
                                save_name = "{0}_{1}".format("nega", negative_count)
                                # 保存样本描述信息
                                negative_file.write(
                                    "{0}.jpg 0 0 0 0 0\n".format(save_name))
                                # 保存样本图片
                                crop_img.save("{0}/{1}.jpg".format(save_img_path, save_name))

                            elif (iou > 0.4) & (iou < 0.65):  # 部分样本
                                part_count += 1
                                save_name = "{0}_{1}".format("part", part_count)
                                # 保存样本描述信息
                                part_file.write(
                                    "{0}.jpg 2 {1} {2} {3} {4}\n".format(save_name, offset_x1, offset_y1, offset_x2,
                                                                         offset_y2))
                                # 保存样本图片
                                crop_img.save("{0}/{1}.jpg".format(save_img_path, save_name))

                            elif iou > 0.65:  # 正样本
                                positive_count += 1
                                save_name = "{0}_{1}".format("posi", positive_count)
                                # 保存样本描述信息
                                positive_file.write(
                                    "{0}.jpg 1 {1} {2} {3} {4}\n".format(save_name, offset_x1, offset_y1, offset_x2,
                                                                         offset_y2))
                                # 保存样本图片
                                crop_img.save("{0}/{1}.jpg".format(save_img_path, save_name))

                        # 取n个负样本
                        negative_sample_count = 0
                        while negative_sample_count < 5:

                            # 随机取点
                            side_len = np.random.randint(face_size, min(img_height, img_width) / 2)

                            _x1 = np.random.randint(0, np.maximum(0, img_width - side_len))
                            _y1 = np.random.randint(0, np.maximum(0, img_height - side_len))
                            _x2 = _x1 + side_len
                            _y2 = _y1 + side_len

                            box = np.array([_x1, _y1, _x2, _y2])
                            boxes = np.array([[x1, y1, x2, y2]])

                            iou = util.IOU(box, boxes, isMin=False)

                            if iou < 0.3:
                                # 扣取并缩放图片
                                crop_img = img.crop(box)
                                crop_img = crop_img.resize((face_size, face_size))

                                negative_count += 1
                                save_name = "{0}_{1}".format("nega", negative_count)
                                # 保存样本描述信息
                                negative_file.write(
                                    "{0}.jpg 0 0 0 0 0\n".format(save_name))
                                # 保存样本图片
                                crop_img.save("{0}/{1}.jpg".format(save_img_path, save_name))

                                negative_sample_count += 1  # 计数加1
                except:
                    # 在取负样本的过程中，极个别原始数据由于size比较小，那里的代码可能出现异常。跳过这些数据即可
                    continue
        except:
            traceback.print_exc()

        finally:
            if not positive_file is None:
                positive_file.close()
            if not negative_file is None:
                negative_file.close()
            if not part_file is None:
                part_file.close()


if __name__ == '__main__':
    sampling = Sampling()
    # sampling.sample(24)
    sampling.sample(48)
    # sampling.sample(12)
    print("Mission Finished!")
