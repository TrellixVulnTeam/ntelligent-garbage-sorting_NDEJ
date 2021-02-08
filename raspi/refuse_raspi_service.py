import cv2
import time
import socket
import json
import sys
import threading
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from steering_engine import control_table

background_print_path = './test image&&video/background_print.png'
video_path = "./test image&&video/1.mp4"


def cv2_img_add_text(img, text, left, top, text_color=(0, 255, 0), text_size=20):
    if isinstance(img, np.ndarray):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    font_text = ImageFont.truetype("font/temp/simsun.ttc", text_size, encoding="utf-8")
    draw.text((left, top), text, text_color, font=font_text)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def display_result(res):
    str_1 = '   1'
    str_ok = '   OK!'
    print(res + str_1 + str_ok)
    image_path = background_print_path
    img = cv2.imread(image_path)
    output_message = cv2_img_add_text(img, res + str_1 + str_ok, 200, 300, (255, 0, 0), 50)
#     cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("result", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("result", output_message)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
    

def load_json(label):
    with open("./garbage_classify_rule.json", 'r', encoding='utf-8') as load_f:
        load_dict = json.load(load_f)
    resresult = load_dict[str(label)]
    return resresult


def handle():
    global flag
    global pre_label
    list_load = np.array([0, 0, 0, 0])
    try:
        while True:
            if flag == 0:
                print('播放视频')
                cap = cv2.VideoCapture(video_path)
                while cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        cv2.namedWindow("video", cv2.WINDOW_AUTOSIZE)
                        #cv2.namedWindow("video", cv2.WINDOW_NORMAL)
                        #cv2.setWindowProperty("video", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                        cv2.imshow("video", frame)
                        cv2.waitKey(25)
                    if flag == 1:
                        cap.release()
                        cv2.destroyAllWindows()
                        break
            elif flag == 1:
                label = pre_label
                results = load_json(label)
                gar_type = int(results[0])
#                 print(type(gar_type), gar_type)
                if list_load[gar_type] == 1:
                    display_result((str(gar_type)+' 号垃圾桶已满'))
                elif list_load[gar_type] == 0:
                    display_result(results)
                load_flag = control_table(gar_type)
                if load_flag == 1:
                    list_load[gar_type] = 1
                else:
                    list_load[gar_type] = 0
                time.sleep(0.05)
    except Exception as exceptions:
        print('主线程错误:', exceptions)


def send_video_receive_results():
    global pre_label
    global flag
    adds = ('192.168.1.100', 8002)
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(adds)
    except socket.error as msg:
        print('线程1错误:', msg)
        sys.exit(1)
    capture = cv2.VideoCapture(0)
    ret, frame = capture.read()
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 99]
    while ret:
        time.sleep(0.01)
        result, img_encode = cv2.imencode('.jpg', frame, encode_param)
        data = np.array(img_encode)
        string_data = data.tostring()
        sock.send(str.encode(str(len(string_data)).ljust(16)))
        sock.send(string_data)
        pre_label = sock.recv(2)
        if len(pre_label):
            pre_label = str(pre_label, encoding='utf-8')
        pre_label = int(pre_label)
#         print(pre_label)
#         if pre_label == 40:
#             flag = 0
        if pre_label != 40:
            flag = 1
            print(flag)
            time.sleep(0.5)
            flag = 0
        ret, frame = capture.read()
        if cv2.waitKey(200) == 27:
            break
    sock.close()


if __name__ == "__main__":
    global flag
    global pre_label
    flag = 0
    pre_label = ''
#     print('**********测试**********')
#     display_result(load_json(9))
#     print('**********测试**********')
#     print('')
    send_receive = threading.Thread(target=send_video_receive_results)
    send_receive.start()
    handle()