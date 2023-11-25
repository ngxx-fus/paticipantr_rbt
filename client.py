# Import socket module
import socket
import cv2
import numpy as np
import torch
import time
import json
import base64
import os

global sendBack_angle, sendBack_Speed, current_speed, current_angle
sendBack_angle = 0
sendBack_Speed = 0
current_speed = 0
current_angle = 0


# Create a socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Define the port on which you want to connect
PORT = 54321
# connect to the server on local computer
s.connect(('127.0.0.1', PORT))

def Control(angle, speed):
    global sendBack_angle, sendBack_Speed
    sendBack_angle = angle
    sendBack_Speed = speed

#-----------------PID Controller-------------------#
error_arr = np.zeros(5)
pre_t = time.time()
MAX_SPEED = 60

def PID(error, p, i, d):
    global pre_t
    # global error_arr
    error_arr[1:] = error_arr[0:-1]
    error_arr[0] = error
    P = error*p
    delta_t = time.time() - pre_t
    #print('DELAY: {:.6f}s'.format(delta_t))
    pre_t = time.time()
    D = (error-error_arr[1])/delta_t*d
    I = np.sum(error_arr)*delta_t*i
    angle = P + I + D
    if abs(angle)>25:
        angle = np.sign(angle)*25
    return int(angle)


if __name__ == "__main__":
    try:
        while True:
            """
            - Chương trình đưa cho bạn 1 giá trị đầu vào:
                * image: hình ảnh trả về từ xe
                * current_speed: vận tốc hiện tại của xe
                * current_angle: góc bẻ lái hiện tại của xe
            - Bạn phải dựa vào giá trị đầu vào này để tính toán và
            gán lại góc lái và tốc độ xe vào 2 biến:
                * Biến điều khiển: sendBack_angle, sendBack_Speed
                Trong đó:
                    + sendBack_angle (góc điều khiển): [-25, 25]
                        NOTE: ( âm là góc trái, dương là góc phải)
                    + sendBack_Speed (tốc độ điều khiển): [-150, 150]
                        NOTE: (âm là lùi, dương là tiến)
            """

            message = bytes(f"{sendBack_angle} {sendBack_Speed}", "utf-8")
            s.sendall(message)
            data = s.recv(100000)
            
            data_recv = json.loads(data)
            current_angle = data_recv["Angle"]
            current_speed = data_recv["Speed"]
            jpg_original = base64.b64decode(data_recv["Img"])
            jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
            image = cv2.imdecode(jpg_as_np, flags=1)

            print(current_speed, current_angle)
            print(image.shape)            
           

            """ Display images """
            cv2.imshow('Image Original', image)
            # save image
            count = 0
            if os.path.exists("./img")==False:
                os.makedirs("./img")
            image_name = "./img/img_{}.jpg".format(count)
            count += 1
            cv2.imwrite(image_name, image)

            Control(sendBack_angle, sendBack_Speed)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        print('closing socket')
        s.close()
