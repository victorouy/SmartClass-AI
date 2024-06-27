import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import optim, nn
import torch.nn.functional as F
import os
import PIL.Image as Image
import torchvision.transforms as transforms
from variant1 import MultiLayerFCNet 

from sklearn.model_selection import train_test_split


class Pclass(Dataset):
    def __init__(self, X, y):

        # path='dataset-cleaned/'
        # path = os.path.join(path, mode)
        self.allaimges= X
        self.clsLabel= y
        # for idx,cls in enumerate(['angry','engaged','happy','neutral']) :
        #     Cpath=os.path.join(path,cls)

        #     F=os.listdir(Cpath)

        #     for im in F:
        #         self.allaimges.append(os.path.join(Cpath,im))
        #         self.clsLabel.append(idx)

        # img_mean = [0.485, 0.456, 0.406]
        # img_std = [0.229, 0.224, 0.225]
        self.mytransform = transforms.Compose([transforms.Resize(size=(64, 64)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize
                                           ])
    def __len__(self):
        return len(self.allaimges)

    def __getitem__(self, idx):


        Im = self.mytransform(Image.open(self.allaimges[idx]).convert('L'))
        Cls=self.clsLabel[idx]

        return Im,Cls


def loadData():
    path = '../dataset/'
    allaimges = []
    clsLabel = []

    for idx, cls in enumerate(['angry', 'engaged', 'happy', 'neutral']):
        Cpath = os.path.join(path, cls)
        F = os.listdir(Cpath)
        for im in F:
            clsLabel.append(idx)
            allaimges.append(os.path.join(Cpath, im))

    testset = Pclass(allaimges, clsLabel)
    return DataLoader(testset, shuffle=True, num_workers=8, drop_last=True)


def loadimg(answer, idx):
    allaimges = []
    clsLabel = [idx]

    allaimges.append(os.path.join("", answer))

    testset = Pclass(allaimges, clsLabel)
    return DataLoader(testset, shuffle=True, num_workers=8, drop_last=True)


if __name__ == '__main__':

    input_size = 64 * 64  # 1 channel, 64x64 image size
    hidden_size = 50  # Number of hidden units
    output_size = 4  # Number of output classes (Our data set has 4 classes)
    label = ""
    answer = input("Type \"dataset\" to evaluate the full dataset or type \"single\" to evaluate: ")

    if answer == "dataset":
        test_loader = loadData()
    elif answer == "single":
        category = input("What category is your image: ")
        if category == 'angry':
            idx = 0
        elif category == 'engaged':
            idx = 1
        elif category == 'happy':
            idx = 2
        elif category == 'neutral':
            idx = 3
        answer2 = input("Type the path of the file: ")
        test_loader = loadimg(answer2, idx)


    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = MultiLayerFCNet(input_size, hidden_size, output_size)
    model = nn.DataParallel(model)
    model.to(device)
    model.load_state_dict(torch.load('../models/best_model_variant1.pth'))
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for instances, labels in test_loader:
            instances, labels = instances.to(device), labels.to(device)
            output = model(instances)
            _, predicted = torch.max(output.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
        if answer == "single":
            _, predicted = torch.max(output.data, 1)
            for i in range(len(instances)):
               if predicted[i] == 0:
                    print("Predicted class is angry")
               if predicted[i] == 1:
                   print("Predicted class is engaged")
               if predicted[i] == 2:
                   print("Predicted class is happy")
               if predicted[i] == 3:
                   print("Predicted class is neutral")
            label = "Label of image is: " + category

    test_accuracy = 100 * test_correct / test_total
    print(f'Test Accuracy: {test_accuracy:.2f}%')
    print(label)
