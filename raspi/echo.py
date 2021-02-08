import time
import RPi.GPIO as GPIO

trigger_pin = 15
echo_pin = 13

GPIO.setmode(GPIO.BOARD)
GPIO.setup(trigger_pin, GPIO.OUT)
GPIO.setup(echo_pin, GPIO.IN)


def send_trigger_pulse():
    GPIO.output(trigger_pin, True)
    time.sleep(0.0001)
    GPIO.output(trigger_pin, False)


def wait_for_echo(value, timeout):
    count = timeout
    while GPIO.input(echo_pin) != value and count > 0:
        count = count-1


def get_distance():
    send_trigger_pulse()
    wait_for_echo(True, 10000)
    start = time.time()
    wait_for_echo(False, 10000)
    finish = time.time()
    pulse_len = finish-start
    distance_cm = pulse_len*34000/2
    return distance_cm


def get_distance_average(count):
    dis_ave = 0
    for i in range(count):
        dis_ave = dis_ave + get_distance()
    dis_ave = dis_ave/count
    print("cm = %f" % dis_ave)
    return dis_ave


def if_full_load(height):
    if get_distance_average(1) < 25:
        load_flag = 1
    else:
        load_flag = 0
    return load_flag

  
if __name__ == "__main__":
    while True:
#         print("cm = %f" % get_distance_average(5))
        get_distance_average(1)
        time.sleep(1)