import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import optim, nn
import torch.nn.functional as F
import os
import PIL.Image as Image
import torchvision.transforms as transforms


# Function to load CIFAR10 dataset


# class DataloaderName(Dataset):
#     def __init__(self, inputprameters):
#
#         #codes
#
#     def __getitem__(self, index):
#
#         # code
#
#         return output
#
#     def __len__(self):
#         return self.__data.shape[0]


class Pclass(Dataset):
    def __init__(self,mode):

        path='dataset-cleaned/'
        path = os.path.join(path, mode)
        self.allaimges=[]
        self.clsLabel=[]
        for idx,cls in enumerate(['angry','engaged','happy','neutral']) :
            Cpath=os.path.join(path,cls)

            F=os.listdir(Cpath)

            for im in F:
                self.allaimges.append(os.path.join(Cpath,im))
                self.clsLabel.append(idx)

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


class MultiLayerFCNet(nn.Module):
    def __init__(self,input_size, hidden_size, output_size):
        super().__init__()

        # self.fc1 = nn.Linear(input_size, hidden_size)
        # self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.fc3 = nn.Linear(hidden_size, output_size)

        self.layer1=nn.Conv2d(1,32,3,padding=1,stride=1)
        self.B1 = nn.BatchNorm2d(32)
        self.layer2 = nn.Conv2d(32, 32, 3, padding=1, stride=1)
        self.B2 = nn.BatchNorm2d(32)
        self.Maxpool=nn.MaxPool2d(2)
        self.layer3 = nn.Conv2d(32, 64, 3, padding=1, stride=1)
        self.B3 = nn.BatchNorm2d(64)
        self.layer4 = nn.Conv2d(64, 64, 3, padding=1, stride=1)
        self.B4 = nn.BatchNorm2d(64)

        self.layer5 = nn.Conv2d(64, 128, 3, padding=1, stride=1)
        self.B5 = nn.BatchNorm2d(128)

        self.layer6 = nn.Conv2d(128, 256, 3, padding=1, stride=1)
        self.B6 = nn.BatchNorm2d(256)



        self.fc = nn.Linear(4 * 4 * 256, output_size)

    def forward(self, x):
        # x = F.relu(self.fc1(x.view(x.size(0),-1)))
        # x = F.relu(self.fc2(x))
        # return F.log_softmax(self.fc3(x), dim=1)


        x = self.B1(F.leaky_relu(self.layer1(x)))
        x =  self.Maxpool(F.leaky_relu(self.layer2(x)))
        x=self.B2(x)
        x=self.B3(self.Maxpool(F.leaky_relu(self.layer3(x))))
        x = self.B4(self.Maxpool(F.leaky_relu(self.layer4(x))))


        x = self.B5((F.leaky_relu(self.layer5(x))))
        x = self.B6(self.Maxpool(F.leaky_relu(self.layer6(x))))


        return self.fc(x.view(x.size(0),-1))

if __name__ == '__main__':

    batch_size = 64
    test_batch_size = 64
    input_size = 64 * 64  # 1 channel, 64x64 image size
    hidden_size = 50  # Number of hidden units
    output_size = 4  # Number of output classes (Our data set has 4 classes)
    epochs = 20


    trainset=Pclass('train')
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)

    testset = Pclass('test')
    test_loader = DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=8, drop_last=True)



    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = MultiLayerFCNet(input_size, hidden_size, output_size)
    model = nn.DataParallel(model)
    model.to(device)
    #model.load_state_dict(torch.load('path'))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    BestACC=0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for instances, labels in train_loader:
            optimizer.zero_grad()
            instances, labels = instances.to(device), labels.to(device)
            output = model(instances)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}')

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0

            for instances, labels in test_loader:
                instances, labels = instances.to(device), labels.to(device)
                output = model(instances)
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()


            ACC = 100 * correct / total
            print(f'Accuracy: {ACC:.2f}%')
            if ACC > BestACC:
                BestACC = ACC
                torch.save(model.state_dict(), 'best_model.pth')
                # torch.save(model.state_dict())
                # torch.save(model.state_dict(), 'path')
        model.train()




