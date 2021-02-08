import os
import ssl
import cv2
import numba
import time
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from numba import jit
from keras.utils.data_utils import get_file
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# ssl._create_default_https_context = ssl._create_unverified_context # 下载模型的时候不想进行ssl证书校验
# 初始化VGG16模型 include_top=False 是指不保留顶层的3个全连接网络层
modelsim = VGG16(weights='imagenet', include_top=False)
WEIGHTS_PATH = './models/vgg16_weights_tf_dim_ordering_tf_kernels.h5'


def get_feature(path):
    # VGG16模型抽取图片特征
    # 图片转成PIL的Image对象，并且对图片做了缩放
    img = image.load_img(path, target_size=(224, 224))
    # 图片转成矩阵、并扩充了维度、最后是预处理
    predict_img = preprocess_input(np.expand_dims(image.img_to_array(img), 0))
    # 丢入vgg16网络做特征抽取，最后返回特征并展平成一维向量方便计算余弦相似度,flatten是numpy.ndarray.flatten的一个函数，即返回一个一维数组。
    return modelsim.predict(predict_img).flatten()


def cos_sim(a, b):
    # 计算两个向量之间的余弦相似度
    a = np.mat(a)
    b = np.mat(b)
    return float(a * b.T) / (np.linalg.norm(a) * np.linalg.norm(b))


def similarity(jpg1, jpg2):
    image_path = [jpg1, jpg2]
    ft1, ft2 = [get_feature(p) for p in image_path]
    res = cos_sim(ft1, ft2)
    return res


if __name__ == '__main__':
    c = 0
    while True:
        path1 = './img_1.jpg'
        path2 = './img_2.jpg'
        print(similarity(path1, path2))
        time1 = time.localtime()
        print(time1.tm_sec)
        print(c)
        c += 1
