import tensorflow as tf
import cv2
import sys
import time
import socket
import threading
import re
import os
import json
import numpy as np
from tools.data_gen import preprocess_img, preprocess_img_from_Url
from models.resnet50 import ResNet50
from keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.models import Model
from keras import regularizers
from similarityf import similarity
# from client import receive_video


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.2
sess = tf.Session(config=config)
# 全局配置文件
tf.app.flags.DEFINE_integer('num_classes', 40, '垃圾分类数目')
tf.app.flags.DEFINE_integer('input_size', 224, '模型输入图片大小')
tf.app.flags.DEFINE_integer('batch_size', 16, '图片批处理大小')
FLAGS = tf.app.flags.FLAGS
h5_weights_path = './output_model/best.h5'
background_path = './background/background.jpg'
foreground_path = './background/subtract.jpg'
background_print = './test image&&video/background_print.png'
s0 = '取背景.............'
s1 = '等待垃圾投放.........'
s2 = '等待投入垃圾稳定......'


def add_new_last_layer(base_model, num_classes):    # 增加最后输出层
    x = base_model.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dropout(0.5, name='dropout1')(x)
    # x = Dense(1024, activation='relu', kernel_regularizer = regularizers.l2(0.0001), name='fc1')(x)
    # x = BatchNormalization(name='bn_fc_00')(x)
    x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0001), name='fc2')(x)
    x = BatchNormalization(name='bn_fc_01')(x)
    x = Dropout(0.5, name='dropout2')(x)
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model


def model_fn(FLAGS):    # 加载模型
    # K.set_learning_phase(0)
    # setup model
    base_model = ResNet50(weights="imagenet",
                          include_top=False,
                          pooling=None,
                          input_shape=(FLAGS.input_size, FLAGS.input_size, 3),
                          classes=FLAGS.num_classes)
    for layer in base_model.layers:
        layer.trainable = False
    model = add_new_last_layer(base_model, FLAGS.num_classes)
    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def init_artificial_neural_network():    # 暴露模型初始化
        model = model_fn(FLAGS)
        model.load_weights(h5_weights_path, by_name=True)
        return model


def prediction_result_from_img(model, img_url):    # 测试图片
    # 加载分类数据
    with open("./garbage_classify_rule.json", 'r', encoding='utf-8') as load_f:
        load_dict = json.load(load_f)
    if re.match(r'^https?:/{2}\w.+$', img_url):
        test_data = preprocess_img_from_Url(img_url, FLAGS.input_size)
    else:
        test_data = preprocess_img(img_url, FLAGS.input_size)
    tta_num = 5
    predictions = [0 * tta_num]
    for i in range(tta_num):
        x_test = test_data[i]
        x_test = x_test[np.newaxis, :, :, :]
        prediction = model.predict(x_test)[0]
        predictions += prediction
    pred_label = np.argmax(predictions, axis=0)
    print('垃圾分类结果:', '  ', pred_label, ',', load_dict[str(pred_label)])
    print('')
    return pred_label, load_dict[str(pred_label)]


def fsm():
    global flag
    global picture
    global name
    global pre_label
    before_state = s0
    state = s0
    model = init_artificial_neural_network()
    before_picture = picture
    print(state)
    while True:
        try:
            flag = 0
            now_picture = picture
            name += 1
            if name == 20:
                name = 0
            sim = similarity(before_picture, now_picture)
            if state == s0:
                if name == 2:
                    name = 0
                    state = s1
                    img_temp = cv2.imread(now_picture)
                    cv2.imwrite(background_path, img_temp)
                else:
                    state = s0
            elif state == s1:
                if sim <= 0.80:
                    state = s2
                else:
                    state = s1
            elif state == s2:
                if sim >= 0.90:
                    print('识别打印')
                    print('')
                    state = s0
                    pre_label, result_model = prediction_result_from_img(model, now_picture)
                    flag = 1
                    name = 0
                    time.sleep(13)
                    before_picture = picture
                    continue
                else:
                    state = s2
            # elif state == s3:
            #     state = s0
            else:
                state = s0
            if state != before_state:
                print(state)
                print('')
            before_state = state
            before_picture = now_picture
        except Exception as e:
            print('异常-fsm:', e)
            continue


def video_receive():
    global flag
    global picture
    global name
    global pre_label
    try:
        address = ('0.0.0.0', 8002)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(address)
        s.listen(1)

        def recv_all(sock, count):
            buf = b''  # buf是一个byte类型
            while count:
                new_buf = sock.recv(count)
                if not new_buf:
                    return None
                buf += new_buf
                count -= len(new_buf)
            return buf

        conn, addr = s.accept()
        print('connect from:' + str(addr))
        while True:
            length = recv_all(conn, 16)  # 获得图片文件的长度,16代表获取长度
            string_data = recv_all(conn, int(length))  # 根据获得的文件长度，获取图片文件
            data = np.frombuffer(string_data, np.uint8)  # 将获取到的字符流数据转换成1维数组
            dec_img = cv2.imdecode(data, cv2.IMREAD_COLOR)  # 将数组解码成图像
            cv2.imshow('SERVER', dec_img)  # 显示图像
            filename = './verify/' + str(name) + '.jpg'
            cv2.imwrite(filename, dec_img)
            picture = filename
            # 发送结果给树莓派
            if flag == 0:
                conn.send(bytes(str(int(40)), encoding='utf-8'))
            else:
                conn.send(bytes(str(int(pre_label)), encoding='utf-8'))
                time.sleep(0.1)
                flag = 0

            k = cv2.waitKey(40) & 0xff
            if k == 27:
                s.close()
                cv2.destroyAllWindows()
                sys.exit(1)
    except Exception as exceptions:
        print('线程1错误:', exceptions)


if __name__ == "__main__":
    global flag
    global pre_label
    global picture
    global name
    flag = 0
    picture = './verify/1.jpg'
    name = 0
    pre_label = 0
    receive = threading.Thread(target=video_receive)
    receive.start()
    fsm()
