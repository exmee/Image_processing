import random
import cv2
import numpy as np
import math
from skimage import exposure
from skimage import util


def _crop_img_bboxes(img, bboxes):
    ''' 图像裁剪
    裁剪后图片要包含所有的框
    输入：
        img：图像array
        bboxes：该图像包含的所有boundingboxes，一个list，每个元素为[x_min,y_min,x_max,y_max]
                要确保是数值
    输出：
        crop_img：裁剪后的图像array
        crop_bboxes：裁剪后的boundingbox的坐标，list
    '''
    # ------------------ 裁剪图像 ------------------
    w = img.shape[1]
    h = img.shape[0]

    x_min = w
    x_max = 0
    y_min = h
    y_max = 0
    for bbox in bboxes:
        x_min = min(x_min, bbox[0])
        y_min = min(y_min, bbox[1])
        x_max = max(x_max, bbox[2])
        y_max = max(y_max, bbox[3])
        name = bbox[4]

    # 包含所有目标框的最小框到各个边的距离
    d_to_left = x_min
    d_to_right = w - x_max
    d_to_top = y_min
    d_to_bottom = h - y_max

    # 随机扩展这个最小范围
    crop_x_min = int(x_min - random.uniform(0, d_to_left))
    crop_y_min = int(y_min - random.uniform(0, d_to_top))
    crop_x_max = int(x_max + random.uniform(0, d_to_right))
    crop_y_max = int(y_max + random.uniform(0, d_to_bottom))

    # 确保不出界
    crop_x_min = max(0, crop_x_min)
    crop_y_min = max(0, crop_y_min)
    crop_x_max = min(w, crop_x_max)
    crop_y_max = min(h, crop_y_max)

    crop_img = img[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

    # ------------------ 裁剪bounding boxes ------------------
    crop_bboxes = list()
    for bbox in bboxes:
        crop_bboxes.append([bbox[0] - crop_x_min, bbox[1] - crop_y_min,
                            bbox[2] - crop_x_min, bbox[3] - crop_y_min, name])

    return crop_img, crop_bboxes

def _shift_pic_bboxes(img, bboxes):
    ''' 图像平移
    平移后需要包含所有的框
    参考资料：https://blog.csdn.net/sty945/article/details/79387054
    输入：
        img：图像array
        bboxes：该图像包含的所有boundingboxes，一个list，每个元素为[x_min,y_min,x_max,y_max]
                要确保是数值
    输出：
        shift_img：平移后的图像array
        shift_bboxes：平移后的boundingbox的坐标，list
    '''
    # ------------------ 平移图像 ------------------
    w = img.shape[1]
    h = img.shape[0]

    x_min = w
    x_max = 0
    y_min = h
    y_max = 0
    for bbox in bboxes:
        x_min = min(x_min, bbox[0])
        y_min = min(y_min, bbox[1])
        x_max = max(x_max, bbox[2])
        y_max = max(x_max, bbox[3])
        name = bbox[4]

    # 包含所有目标框的最小框到各个边的距离，即每个方向的最大移动距离
    d_to_left = x_min
    d_to_right = w - x_max
    d_to_top = y_min
    d_to_bottom = h - y_max

    # 在矩阵第一行中表示的是[1,0,x],其中x表示图像将向左或向右移动的距离，如果x是正值，则表示向右移动，如果是负值的话，则表示向左移动。
    # 在矩阵第二行表示的是[0,1,y],其中y表示图像将向上或向下移动的距离，如果y是正值的话，则向下移动，如果是负值的话，则向上移动。
    x = random.uniform(-(d_to_left / 3), d_to_right / 3)
    y = random.uniform(-(d_to_top / 3), d_to_bottom / 3)
    M = np.float32([[1, 0, x], [0, 1, y]])

    # 仿射变换
    shift_img = cv2.warpAffine(img, M,
                               (img.shape[1], img.shape[0]))  # 第一个参数表示我们希望进行变换的图片，第二个参数是我们的平移矩阵，第三个希望展示的结果图片的大小

    # ------------------ 平移boundingbox ------------------
    shift_bboxes = list()
    for bbox in bboxes:
        shift_bboxes.append([bbox[0] + x, bbox[1] + y, bbox[2] + x, bbox[3] + y, name])

    return shift_img, shift_bboxes

def _changeLight(img):
    ''' 改变亮度
    adjust_gamma(image, gamma=1, gain=1)函数:
    gamma>1时，输出图像变暗，小于1时，输出图像变亮
    输入：
        img：图像array
    输出：
        img：改变亮度后的图像array
    '''
    flag = random.uniform(0.5, 1.5)  ##flag>1为调暗,小于1为调亮
    return exposure.adjust_gamma(img, flag)

def _addNoise(img):
    ''' 加入噪声
    输入：
        img：图像array
    输出：
        img：加入噪声后的图像array,由于输出的像素是在[0,1]之间,所以得乘以255
    '''
    return util.random_noise(img, mode='gaussian', clip=True) * 255


def _rotate_img_bboxes(img, bboxes, angle=5, scale=1.):
    ''' 图像旋转
    参考：https://blog.csdn.net/saltriver/article/details/79680189
          https://www.ctolib.com/topics-44419.html
    关于仿射变换：https://www.zhihu.com/question/20666664
    输入:
        img:图像array,(h,w,c)
        bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
        angle:旋转角度
        scale:默认1
    输出:
        rot_img:旋转后的图像array
        rot_bboxes:旋转后的boundingbox坐标list
    '''
    # ---------------------- 旋转图像 ----------------------
    w = img.shape[1]
    h = img.shape[0]
    # 角度变弧度
    rangle = np.deg2rad(angle)
    # 计算新图像的宽度和高度，分别为最高点和最低点的垂直距离
    nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
    nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
    # 获取图像绕着某一点的旋转矩阵
    # getRotationMatrix2D(Point2f center, double angle, double scale)
    # Point2f center：表示旋转的中心点
    # double angle：表示旋转的角度
    # double scale：图像缩放因子
    # 参考：https://cloud.tencent.com/developer/article/1425373
    rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)  # 返回 2x3 矩阵
    # 新中心点与旧中心点之间的位置
    rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]
    # 仿射变换
    rot_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))),
                             flags=cv2.INTER_LANCZOS4)  # ceil向上取整

    # ---------------------- 矫正boundingbox ----------------------
    # rot_mat是最终的旋转矩阵
    # 获取原始bbox的四个中点，然后将这四个点转换到旋转后的坐标系下
    rot_bboxes = list()
    for bbox in bboxes:
        x_min = bbox[0]
        y_min = bbox[1]
        x_max = bbox[2]
        y_max = bbox[3]
        name = bbox[4]
        point1 = np.dot(rot_mat, np.array([(x_min + x_max) / 2, y_min, 1]))
        point2 = np.dot(rot_mat, np.array([x_max, (y_min + y_max) / 2, 1]))
        point3 = np.dot(rot_mat, np.array([(x_min + x_max) / 2, y_max, 1]))
        point4 = np.dot(rot_mat, np.array([x_min, (y_min + y_max) / 2, 1]))

        # 合并np.array
        concat = np.vstack((point1, point2, point3, point4))  # 在竖直方向上堆叠
        # 改变array类型
        concat = concat.astype(np.int32)
        # 得到旋转后的坐标
        rx, ry, rw, rh = cv2.boundingRect(concat)
        rx_min = rx
        ry_min = ry
        rx_max = rx + rw
        ry_max = ry + rh
        # 加入list中
        rot_bboxes.append([rx_min, ry_min, rx_max, ry_max, name])

    return rot_img, rot_bboxes


def _flip_pic_bboxes(img, bboxes):
    ''' 图像镜像
    参考：https://blog.csdn.net/jningwei/article/details/78753607
    镜像后的图片要包含所有的框
    输入：
        img：图像array
        bboxes：该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
    输出:
        flip_img:镜像后的图像array
        flip_bboxes:镜像后的bounding box的坐标list
    '''
    # ---------------------- 镜像图像 ----------------------
    import copy
    flip_img = copy.deepcopy(img)
    if random.random() < 0.5:
        horizon = True
    else:
        horizon = False
    h, w, _ = img.shape
    if horizon:  # 水平翻转
        flip_img = cv2.flip(flip_img, -1)
    else:
        flip_img = cv2.flip(flip_img, 0)
    # ---------------------- 矫正boundingbox ----------------------
    flip_bboxes = list()
    for bbox in bboxes:
        x_min = bbox[0]
        y_min = bbox[1]
        x_max = bbox[2]
        y_max = bbox[3]
        name = bbox[4]
        if horizon:
            flip_bboxes.append([w - x_max, y_min, w - x_min, y_max, name])
        else:
            flip_bboxes.append([x_min, h - y_max, x_max, h - y_min, name])

    return flip_img, flip_bboxes

if __name__ == '__main__':
    pass
    #_crop_img_bboxes(img, bboxes)