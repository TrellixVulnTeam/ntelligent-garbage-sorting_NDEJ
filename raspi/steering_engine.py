from __future__ import division
import time
import Adafruit_PCA9685
from echo import if_full_load

pwm = Adafruit_PCA9685.PCA9685()  # 把Adafruit_PCA9685.PCA9685()引用地址赋给PWM标签
pwm.set_pwm_freq(50)
channel_table = 2
channel_door = 0

date_forward2 = int(4096 * 1200 / 20000)
date_reversal2 = int(4096 * 1800 / 20000)
date_forward3 = int(4096 * 900 / 20000)
date_reversal3 = int(4096 * 2100 / 20000)
date_forward4 = int(4096 * 700 / 20000)
date_reversal4 = int(4096 * 2300 / 20000)
time_angle_ninety = 1.3


def set_servo_angle(channel, angle):
    date = int(4096 * (int(angle * 11.11111) + 500) / 20000)
    pwm.set_pwm(channel, 0, date)


def stop_table():
    pwm.set_pwm(channel_table, 0, int(4096*1500/20000))
# int(4096 * 1500 / 20000)

def control_table(label_gar):
    label_gar = label_gar
    if label_gar == 2:    #kehuishou
        pwm.set_pwm(channel_table, 0, date_forward2)
        time.sleep(1.50)
        stop_table()
        res = control_door()
        pwm.set_pwm(channel_table, 0, date_reversal2)
        time.sleep(1.35)
        stop_table()
    elif label_gar == 3:    #chuyu
        pwm.set_pwm(channel_table, 0, date_forward3)
        time.sleep(2.15)
        stop_table()
        res = control_door()
        pwm.set_pwm(channel_table, 0, date_reversal3)
        time.sleep(2.05)
        stop_table()
    elif label_gar == 4:    #qitalaji
        pwm.set_pwm(channel_table, 0, date_forward4)
        time.sleep(2.85)
        stop_table()
        res = control_door()
        pwm.set_pwm(channel_table, 0, date_reversal4)
        time.sleep(2.85)
        stop_table()   
    elif label_gar == 1:  #youhailaji
        res = control_door()
    return res


def init_door():
    set_servo_angle(channel_door, 1)


def control_door():
    set_servo_angle(channel_door, 180)
    time.sleep(1)
    loader = if_full_load(70)
    print(loader)
    set_servo_angle(channel_door, 5)
    time.sleep(1)
    return loader


if __name__ == "__main__":
#     set_servo_angle(channel_door, 1)
#     time.sleep(0.5)
    print('Moving servo on channel x, press Ctrl-C to quit...')
    while True:
        print('label of bar')
        control_table(int(input()))
#     pwm.set_pwm(channel_table, 0, int(4096*1200/20000))
#     time.sleep(time_angle_ninety)
#     pwm.set_pwm(channel_table, 0, int(4096*1500/20000))
#     time.sleep(2)
#     pwm.set_pwm(channel_table, 0, int(4096*1800/20000))
#     time.sleep(time_angle_ninety)
#     pwm.set_pwm(channel_table, 0, int(4096*1500/20000))
