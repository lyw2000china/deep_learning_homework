import torch
from torchvision import datasets, transforms
import os

batch_size = 32
timesteps = 1000

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

traindir = os.path.join('realdata', 'train_ImageFolder')
valdir = os.path.join('realdata', 'val_ImageFolder')
train = datasets.ImageFolder(traindir, transform=transform)
val = datasets.ImageFolder(valdir, transform=transform)
train_loader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)