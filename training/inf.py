import cv2
import numpy as np
import time
from PIL import Image
import torch
from torchvision import models
import torch.nn as nn
import torchvision.transforms as transforms


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()


model = models.alexnet(pretrained=False)
model.classifier[6] = nn.Linear(4096, 26)  # replace with your actual class count
model.load_state_dict(torch.load("alexnet_asl.pth", map_location=torch.device('cpu')))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

while True:
    # 逐帧捕获
    ret, frame = cap.read()
    cv2.imshow('Camera', frame)
    
    if not ret:
        print("无法接收帧，退出 ...")
        break

    resized = cv2.resize(frame, (224, 224))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    img = Image.fromarray(rgb)  # ← 关键步骤：转为 PIL.Image
    input_tensor = transform(img)  # shape: [3, 224, 224]
    input_tensor = input_tensor.unsqueeze(0)  # shape: [1, 3, 224, 224]

    with torch.no_grad():
        outputs = model(input_tensor.to(device))
        predicted_class = outputs.argmax(dim=1).item()
        print(chr(predicted_class+ord('A')))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    time.sleep(0.5)

# 释放资源
cap.release()
cv2.destroyAllWindows()
