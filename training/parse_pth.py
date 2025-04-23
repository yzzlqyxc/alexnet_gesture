import torch
import torchvision.transforms as transforms
from train import Dwarf


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Dwarf(num_classes=10).to(device)
model.load_state_dict(torch.load("dwarf.pth", map_location=torch.device('cpu')))
model.to(device)
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

checkpoint = torch.load("dwarf.pth")
state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

#print(model.named_parameters())
with open("weights.txt", "w") as f:

    for name, param in model.named_parameters():
        cnt = 0
        for i in param.shape:
            f.write(f"{i} ")
        f.write('\n');
        flat = param.flatten()
        for i in flat: 
            f.write(f"{i.item():.10f} ")
            cnt += 1
        f.write('\n');


