from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch
import torch.nn as nn
import torch.nn.functional as F

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])


class Dwarf(nn.Module):
    def __init__(self, num_classes=10):
        super(Dwarf, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 10)

        self.pool  = nn.MaxPool2d(2, 2) 
        self.pool2  = nn.MaxPool2d(8, 8) 

    def forward(self, x):
        x = F.relu(self.conv1(x))  
        
        x = self.conv2(x)
        x = F.relu(self.pool(x))

        x = self.conv3(x)
        x = F.relu(self.pool(x))

        x = self.conv4(x)
        x = F.relu(self.pool2(x))

        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))  
        x = self.fc2(x)          
        return x


if __name__ == '__main__': 
    dataset = datasets.ImageFolder(root='./data/Dataset', transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Dwarf(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计信息
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")

    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**2,1), 'MB')
    print('Reserved: ', round(torch.cuda.memory_reserved(0)/1024**2,1), 'MB')

    torch.save(model.state_dict(), "dwarf.pth")