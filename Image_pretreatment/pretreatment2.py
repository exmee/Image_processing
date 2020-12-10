from PIL import Image, ImageEnhance, ImageOps, ImageFile
import numpy as np
import random
import threading, os, time
import logging
from skimage import exposure
from skimage import util
import cv2

logger = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True

'''
class DataAugmentation:

    def __init__(self):
        pass

    @staticmethod
    def openImage(image):
        return Image.open(image, mode="r")

    @staticmethod
    def randomRotation(image, mode=Image.BICUBIC):
        """
         对图像进行随机任意角度(0~360度)旋转
        :param mode 邻近插值,双线性插值,双三次B样条插值(default)
        :param image PIL的图像image
        :return: 旋转转之后的图像
        """
        random_angle = np.random.randint(1, 360)
        return image.rotate(random_angle, mode)

    @staticmethod
    def randomCrop(image):
        """
        对图像随意剪切,考虑到图像大小范围(68,68),使用一个一个大于(36*36)的窗口进行截图
        :param image: PIL的图像image
        :return: 剪切之后的图像
        """
        image_width = image.size[0]
        image_height = image.size[1]
        crop_win_size = np.random.randint(40, 68)
        random_region = (
            (image_width - crop_win_size) >> 1, (image_height - crop_win_size) >> 1, (image_width + crop_win_size) >> 1,
            (image_height + crop_win_size) >> 1)
        return image.crop(random_region)

    @staticmethod
    def randomColor(image):
        """
        对图像进行颜色抖动
        :param image: PIL的图像image
        :return: 有颜色色差的图像image
        """
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
        random_factor = np.random.randint(10, 21) / 10.  # 随机因子
        brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
        random_factor = np.random.randint(10, 21) / 10.  # 随机因1子
        contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度

    @staticmethod
    def randomGaussian(image, mean=0.2, sigma=0.3):
        """
         对图像进行高斯噪声处理
        :param image:
        :return:
        """

        def gaussianNoisy(im, mean=0.2, sigma=0.3):
            """
            对图像做高斯噪音处理
            :param im: 单通道图像
            :param mean: 偏移量
            :param sigma: 标准差
            :return:
            """
            for _i in range(len(im)):
                im[_i] += random.gauss(mean, sigma)
            return im

        # 将图像转化成数组
        img = np.asarray(image)
        img.flags.writeable = True  # 将数组改为读写模式
        width, height = img.shape[:2]
        img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)
        img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)
        img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)
        img[:, :, 0] = img_r.reshape([width, height])
        img[:, :, 1] = img_g.reshape([width, height])
        img[:, :, 2] = img_b.reshape([width, height])
        return Image.fromarray(np.uint8(img))

    @staticmethod
    def saveImage(image, path):
        image.save(path)
'''


def openImage(image):
    return Image.open(image, mode="r")


def randomRotation(image, mode=Image.BICUBIC):
    """
     对图像进行随机任意角度(0~360度)旋转
    :param mode 邻近插值,双线性插值,双三次B样条插值(default)
    :param image PIL的图像image
    :return: 旋转转之后的图像
    """
    random_angle = np.random.randint(1, 90)
    return image.rotate(random_angle, mode)


def randomCrop(image):
    """
    对图像随意剪切,考虑到图像大小范围(68,68),使用一个一个大于(36*36)的窗口进行截图
    :param image: PIL的图像image
    :return: 剪切之后的图像
    """
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_size = np.random.randint(1080, 1920)
    random_region = (
        (image_width - crop_win_size) >> 1, (image_height - crop_win_size) >> 1, (image_width + crop_win_size) >> 1,
        (image_height + crop_win_size) >> 1)
    return image.crop(random_region)


def randomColor(image):
    """
    对图像进行颜色抖动
    :param image: PIL的图像image
    :return: 有颜色色差的图像image
    """
    random_factor = np.random.randint(0, 31) / 10.  # 随机因子
    color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
    random_factor = np.random.randint(10, 21) / 10.  # 随机因子
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
    random_factor = np.random.randint(10, 21) / 10.  # 随机因1子
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
    random_factor = np.random.randint(0, 31) / 10.  # 随机因子
    return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度



def randomGaussian(image, mean=0.2, sigma=0.3):
    """
     对图像进行高斯噪声处理
    :param image:
    :return:
    """

    def gaussianNoisy(im, mean=0.2, sigma=0.3):
        """
        对图像做高斯噪音处理
        :param im: 单通道图像
        :param mean: 偏移量
        :param sigma: 标准差
        :return:
        """
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    # 将图像转化成数组
    img = np.asarray(image)
    #img.flags.writeable = True  # 将数组改为读写模式
    width, height = img.shape[:2]
    img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)
    img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)
    img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)
    img[:, :, 0] = img_r.reshape([width, height])
    img[:, :, 1] = img_g.reshape([width, height])
    img[:, :, 2] = img_b.reshape([width, height])
    return Image.fromarray(np.uint8(img))

def saveImage(image, path):
    image.save(path)

if __name__ == '__main__':
    '''
    im_dir = 'D:\\sz4361\\公司项目文件\\舱门检测\\舱门视频\\舱门视频\\test\\images\\frame10.jpg'
    save_path = 'D:\\sz4361\\公司项目文件\\舱门检测\\舱门视频\\舱门视频\\test\\save_imgs\\frame10.jpg'
    img = openImage(im_dir)
    img = randomRotation(img)
    #img = randomCrop(img)
    #img = randomColor(img)
    saveImage(img, save_path)
    '''
    im_dir = 'D:\\sz4361\\公司项目文件\\舱门检测\\舱门视频\\舱门视频\\select\\three\\'  # 帧存放路径
    output_dir = 'D:\\sz4361\\公司项目文件\\舱门检测\\舱门视频\\舱门视频\\select\\recolorthree\\'  # 帧存放路径
    im_list = os.listdir(im_dir)
    for i in im_list:
        im_name = os.path.join(im_dir + i)
        img = openImage(im_name)
        img = randomColor(img)
        saveImage(img, output_dir + "crop" + i)


