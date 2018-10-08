import argparse

# def test():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--t1", dest="__t1", default="123", type=str)
#     parser.add_argument("-t2", default=456., type=float)
#
#     parser.add_argument("--weights", help=
#     "weightsfile", default="yolov3.weights", type=str)
#
#     return parser.parse_args()
#
#
# # args = test()
# # print(args.__t1)
#
# aa = ("1", "2")
# bb = [int(a) for a in aa]
# print(bb)

import numpy as np


# box[x1,y1,x2,y2]
def IOU(box, boxes, isMin=False):
    max_x1 = np.maximum(box[0], boxes[:, 0])
    max_y1 = np.maximum(box[1], boxes[:, 1])

    min_x2 = np.minimum(box[2], boxes[:, 3])
    min_y2 = np.minimum(box[3], boxes[:, 3])

    across_w = np.maximum(0, min_x2 - max_x1)
    across_h = np.maximum(0, min_y2 - max_y1)

    across_area = across_w * across_h

    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    if isMin:
        iou = across_area / np.minimum(box_area, boxes_area)
    else:
        iou = across_area / (box_area + boxes_area - across_area)
    return iou


# boxes类型是numpy.ndarray
def NMS(boxes, th=0.5):
    if len(boxes) == 0:
        return np.array([])

    newboxes = sorted(boxes, key=lambda x: x[4], reverse=True)  # 处理后newboxes类型会变成list
    newboxes = np.stack(newboxes)
    list = []
    while len(newboxes) > 1:
        max_box = newboxes[0]
        list.append(max_box)
        rest_boxes = newboxes[1:]

        iou = IOU(max_box, rest_boxes)
        newboxes = rest_boxes[np.where(iou < th)]

    if len(newboxes) == 1:
        list.append(newboxes[0])
    return np.stack(list)


import torch

if __name__ == '__main__':
    box = np.array([1, 1, 3, 3])
    boxes = np.array([[2, 2, 6, 7]])
    # print(IOU(box, boxes, False))

    # print(np.where(box < 2))
    #
    # print(box[box < 0])

    boxes = [[1, 1, 3, 3, 0.97], [0, 2, 2, 4, 0.8], [2, 2, 6, 7, 0.85], [3, 3, 6, 7, 0.82]]
    print(NMS(np.array(boxes)))

    bb = np.array([[1], [7], [3], [5], [4]])
    cc = np.array([[1, 2, 3, 4], [7, 2, 3, 4], [3, 2, 3, 4], [5, 2, 3, 4], [4, 2, 3, 4]])

    dd = torch.Tensor(cc)
    print(np.where(cc < 3))
    print(torch.nonzero(dd < 3))
