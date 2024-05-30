import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Randomly change brightness, and rotations
data_transforms = transforms.Compose([
    transforms.RandomRotation(10),      # Randomly rotate images by ±10 degrees
    transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Randomly change brightness, contrast, saturation, and hue
    transforms.ToTensor()               # Convert images to PyTorch tensors
])

# PATH OF data set
dataset_dir = 'dataset/test'

# Load dataset with data augmentation
dataset = datasets.ImageFolder(root=dataset_dir, transform=data_transforms)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


data_iter = iter(dataloader)
images, labels = next(data_iter)


