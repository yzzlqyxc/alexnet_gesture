import os
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import alexnet
from sklearn.metrics import accuracy_score
from datasets import load_dataset
import torch.optim as optim

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device.type)

# Data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

ds = load_dataset("Marxulia/asl_sign_languages_alphabets_v03")

# Dataset paths
class ASLHuggingFaceDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]['image']
        label = self.dataset[idx]['label']
        if self.transform:
            image = self.transform(image)
        return image, label

# Load datasets
print(ds)
train_dataset = ASLHuggingFaceDataset(ds['train'], transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Load pretrained AlexNet
model = alexnet(pretrained=True)
print(model)

# Replace the final classifier layer for 26 classes
model.classifier[6] = nn.Linear(model.classifier[6].in_features, 26)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    print(epoch)
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.numpy())

    acc = accuracy_score(val_labels, val_preds)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss:.4f}, Val Acc: {acc:.4f}")

print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**2,1), 'MB')
print('Reserved: ', round(torch.cuda.memory_reserved(0)/1024**2,1), 'MB')

#torch.save(model.state_dict(), "alexnet_asl.pth")
