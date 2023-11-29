import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchmetrics
from torchmetrics import Dice, JaccardIndex
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2 #np.array -> torch.tensor
import os
from tqdm import tqdm
from glob import glob
import socket
import time
import json
import base64
import os



global sendBack_angle, sendBack_Speed, current_speed, current_angle
sendBack_angle = 0
sendBack_Speed = 0
current_speed = 0
current_angle = 0
count = 0

######################### begin of paticipant's setup ###############################
global trainsize_h, trainsize_w, to_degree, to_radian
trainsize_w = 640
trainsize_h = 176
to_degree = 57.2957795
to_radian =  0.01745329
#cd  dxdevided by 16


test_transform = A.Compose([
    A.Resize(width=trainsize_w, height=trainsize_h),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
    ToTensorV2(), # numpy.array -> torch.tensor (B, 3, H, W)
])

# train_dataset = LaneDataset("./", "./annotations/trainval.txt", train_transform)
# test_dataset = LaneDataset("./", "./annotations/test.txt", test_transform)

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

#model UNet
def unet_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        nn.ReLU()
    )

class UNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.downsample = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        self.block_down1 = unet_block(3, 64)
        self.block_down2 = unet_block(64, 128)
        self.block_down3 = unet_block(128, 256)
        self.block_down4 = unet_block(256, 512)
        self.block_neck = unet_block(512, 1024)
        self.block_up1 = unet_block(1024+512, 512)
        self.block_up2 = unet_block(256+512, 256)
        self.block_up3 = unet_block(128+256, 128)
        self.block_up4 = unet_block(128+64, 64)
        self.conv_cls = nn.Conv2d(64, self.n_classes, 1) # -> (B, n_class, H, W)

    def forward(self, x):
        # (B, C, H, W)
        x1 = self.block_down1(x)
        x = self.downsample(x1)
        x2 = self.block_down2(x)
        x = self.downsample(x2)
        x3 = self.block_down3(x)
        x = self.downsample(x3)
        x4 = self.block_down4(x)
        x = self.downsample(x4)

        x = self.block_neck(x)

        x = torch.cat([x4, self.upsample(x)], dim=1)
        x = self.block_up1(x)
        x = torch.cat([x3, self.upsample(x)], dim=1)
        x = self.block_up2(x)
        x = torch.cat([x2, self.upsample(x)], dim=1)
        x = self.block_up3(x)
        x = torch.cat([x1, self.upsample(x)], dim=1)
        x = self.block_up4(x)

        x = self.conv_cls(x)
        return x

class aTest(Dataset):
    def __init__(self, img, transform = None): #transform: augmentation + np.array—> torch. tensor
        super().__init__()
        self.img = img
        self.transform = transform
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        image = self.img
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            transformed = self.transform(image=image)
            transformed_image = transformed['image']
        return transformed_image

######################### end of paticipant's setup ###############################

################################ order part ##################################
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

################## device #######################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
################### model ######################
model = UNet(n_classes=1).to(device)
################### load model ######################
model.load_state_dict(torch.load("./model_ep_v1_30.pth", map_location=device))
model.eval()  #be skipped cause of much running time 
def get_angle(x1,y1, x2,y2):
    to_degree = 180/3.141593
    return math.atan((x1-x2)/(y2-y1))*to_degree

def segmentation(img):
    a_test_loader = aTest(img=img, transform=test_transform)
    with torch.no_grad():
        x = a_test_loader[0]
        x = x.to(device).unsqueeze(0)
        y_hat = model(x).squeeze() # (1, 1, H, W)
        y_hat_mask = y_hat.sigmoid().round().long()
        # plt.imshow(y_hat_mask.cpu() )
        plt.imsave("./pred.png", y_hat_mask.cpu())
        return y_hat_mask.cpu()
    
def get_sendback_angle() :
    img_ = cv2.imread("./pred.png",cv2.IMREAD_COLOR) # road.png is the filename
    gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 200)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 68, minLineLength=15, maxLineGap=250)

    sum_angle  = 0
    angles = 0
    minx = trainsize_w
    maxyx = 0
    miny = trainsize_h
    maxy = 0
    try:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cur_angle = math.atan((x1-x2)/(y2-y1))*to_degree
            if abs(cur_angle) > 70 :
                continue
            sum_angle+=cur_angle
            angles+=1
            cv2.line(img_, (x1, y1), (x2, y2), (255, 0, 0), 3)
    except:
        # sum_angle=10
        angles=10
    cv2.imshow("img_.jpg", img_)
    # cv2.imwrite("img_.jpg", img_)
    if angles == 0 : 
        angles = 1
    return sum_angle/angles


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
            top = 360
            bottom = 180
            left = 0
            right = 640
            image = image[bottom : top , left : right]
            """ Display images """
            cv2.imshow('Image Original', image)
            
            img = segmentation(image)
            
            print("sendBack_angle = ", sendBack_angle)
            sendBack_angle_p = 0.0
            sendBack_angle_p = get_sendback_angle()*0.6
            print("sendBack_angle_p = ", sendBack_angle_p)
            
            if abs(sendBack_angle_p) < 5:
                sendBack_Speed = 10
            elif abs(sendBack_angle_p) < 10:
                sendBack_Speed = 5
            elif abs(sendBack_angle_p) < 15:
                sendBack_Speed = -7
            elif abs(sendBack_angle_p) < 19:
                sendBack_Speed = -7
            Control(sendBack_angle_p, sendBack_Speed)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        print('closing socket')
        s.close()
