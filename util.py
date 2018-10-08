import numpy as np


def IOU(box, boxes, isMin=False):
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # 交集区域(如果相交的话)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    j_w = np.maximum(0, xx2 - xx1)
    j_h = np.maximum(0, yy2 - yy1)

    # 交集区域面积
    j_area = j_w * j_h

    if isMin:
        value = j_area / np.minimum(box_area, boxes_area)
    else:
        value = j_area / (box_area + boxes_area - j_area)

    return value


def NMS(boxes, isMin=False, thresh=0.3):
    if len(boxes) == 0:
        return np.array([])

    # 将boxes降序排列
    # _boxes = boxes[(-boxes[:, 4]).argsort()]
    _boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    _boxes = np.array(_boxes)
    final_boxes = []

    while len(_boxes) > 1:
        max_box = _boxes[0]  # 找出置信度最大的box
        rest_boxes = _boxes[1:]  # 剩余的boxes

        final_boxes.append(max_box)
        index = np.where(
            IOU(max_box, rest_boxes, isMin) < thresh)  # 找出和max_box做IOU运算，其值小于thresh的box在rest_boxes中的下标

        _boxes = rest_boxes[index]  # 找出符合条件的box，循环进行下一次nms运算

    if len(_boxes) == 1:
        final_boxes.append(_boxes[0])

    return np.stack(final_boxes)


# 将box(批量)变为正方形
def convert_to_square(boxes):
    if boxes.shape[0] == 0:
        return np.array([])

    bboxes = boxes
    w = bboxes[:, 2] - bboxes[:, 0]
    h = bboxes[:, 3] - bboxes[:, 1]

    max_side = np.maximum(w, h)

    bboxes[:, 0] = bboxes[:, 0] - (max_side / 2 - w / 2)
    bboxes[:, 1] = bboxes[:, 1] - (max_side / 2 - h / 2)
    bboxes[:, 2] = bboxes[:, 0] + max_side
    bboxes[:, 3] = bboxes[:, 1] + max_side

    return bboxes


# x1,y1,x2,y2
def test_fn(boxes):
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    max_len = np.maximum(w, h)
    # 中心点坐标
    c_x = (boxes[:, 0] + boxes[:, 2]) / 2
    c_y = (boxes[:, 1] + boxes[:, 3]) / 2

    new_x1 = c_x - max_len / 2
    new_y1 = c_y - max_len / 2

    new_x2 = new_x1 + max_len
    new_y2 = new_y1 + max_len

    boxes[:, 0] = new_x1
    boxes[:, 1] = new_y1
    boxes[:, 2] = new_x2
    boxes[:, 3] = new_y2

    return boxes


if __name__ == '__main__':
    # a = np.array(([[1, 2, 3, 4, 5]]))
    # print(np.minimum(2, a))

    a = np.array([[1, 3, 2, 6], [2, 6, 4, 9]])
    b = np.array([[1, 3, 2, 6], [2, 6, 4, 9]])

    print(convert_to_square(a))
    print(a)
    print(test_fn(b))
