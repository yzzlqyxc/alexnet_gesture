import cv2
import numpy as np
import time
from PIL import Image
import torch
from torchvision import models
import torch.nn as nn
import torchvision.transforms as transforms
from train import Dwarf


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Dwarf(num_classes=10).to(device)

model.load_state_dict(torch.load("dwarf.pth", map_location=torch.device('cpu')))
model.eval()
model.to(device)

while True:
    # 逐帧捕获
    ret, frame = cap.read()
    cv2.imshow('Camera', frame)
    
    if not ret:
        print("无法接收帧，退出 ...")
        break

    resized = cv2.resize(frame, (32, 32))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    img = Image.fromarray(rgb) 
    input_tensor = transform(img) 
    input_tensor = input_tensor.unsqueeze(0)  

    with torch.no_grad():
        outputs = model(input_tensor.to(device))
        predicted_class = outputs.argmax(dim=1).item()
        print(predicted_class)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    time.sleep(1)

# 释放资源
cap.release()
cv2.destroyAllWindows()