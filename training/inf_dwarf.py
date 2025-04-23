import cv2
import numpy as np
import time
from PIL import Image
import torch
from torchvision import models
import torch.nn as nn
import torchvision.transforms as transforms
from train import Dwarf
from pathlib import Path


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Dwarf(num_classes=10).to(device)
model.load_state_dict(torch.load("dwarf.pth", map_location=torch.device('cpu')))
model.to(device)
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def inf_fromtenser(img):
    model.eval()

    img = Image.fromarray(img) 
    input_tensor = transform(img) 
    input_tensor = input_tensor.unsqueeze(0)  

    with torch.no_grad():
        outputs = model(input_tensor.to(device))
        predicted_class = outputs.argmax(dim=1).item()
        return predicted_class

def camera_inf():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        exit()

    while True:
        # 逐帧捕获
        ret, frame = cap.read()
        
        if not ret:
            print("无法接收帧，退出 ...")
            break

        resized = cv2.resize(frame, (32, 32))
        cv2.imshow('Camera', resized)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        print(inf_fromtenser(rgb))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.1)

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

def pic_inf(pic_dir : str):
    img = cv2.imread(pic_dir)
    img = cv2.resize(img, (32, 32))

    return inf_fromtenser(img)

# pic_inf("./data/Dataset/0/IMG_1118.JPG")

camera_inf()

for i in range(0, 10):
    a = './data/Dataset/' + str(i)
    path = Path(a)
    cnt, cntb, total = 0, 0, 0
    for file in path.rglob('*'):  # rglob() 用于递归查找所有文件和子文件夹
        a = pic_inf(file)
        # print(file, pic_inf(file))
        if a == i :
            cnt += 1
        elif a == 5:
            cntb += 1
        total += 1
    print(f"{i} accuray: {cnt/total}, {total} {cntb}")
