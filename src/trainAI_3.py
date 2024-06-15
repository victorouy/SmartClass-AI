import torch
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import optim, nn
import torch.nn.functional as F
import os
import PIL.Image as Image
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split


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


def fetchData2():
    path = '../dataset-cleaned/'
    allaimges = []
    clsLabel = []

    for idx, cls in enumerate(['angry', 'engaged', 'happy', 'neutral']):
        Cpath = os.path.join(path, cls)
        F = os.listdir(Cpath)
        for im in F:
            allaimges.append(os.path.join(Cpath, im))
            clsLabel.append(idx)

    # Split data into 70% train and 30% temporary (to be split further into validation and test)
    X_train, X_temp, y_train, y_temp = train_test_split(allaimges, clsLabel, test_size=0.30, random_state=42, stratify=clsLabel)

    # Split the temporary set into 50% validation and 50% test (15% each of the total dataset)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

    return X_train, X_valid, X_test, y_train, y_valid, y_test



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
        self.dropout = nn.Dropout(0.5)





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
        x = x.view(x.size(0), -1)
        x = self.dropout(x)


        return self.fc(x)

if __name__ == '__main__':

    batch_size = 64
    test_batch_size = 64
    input_size = 64 * 64  # 1 channel, 64x64 image size
    hidden_size = 50  # Number of hidden units
    output_size = 4  # Number of output classes (Our data set has 4 classes)
    epochs = 20
    patience = 3


    # X_train, X_valid, X_test, y_train, y_valid, y_test = fetchData2()
    with open('../dataset_splits.pkl', 'rb') as f:
        splits = pickle.load(f)
    X_train = splits['X_train']
    X_valid = splits['X_valid']
    X_test = splits['X_test']
    y_train = splits['y_train']
    y_valid = splits['y_valid']
    y_test = splits['y_test']

    trainset=Pclass(X_train, y_train)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)

    testset = Pclass(X_test, y_test)
    test_loader = DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=8, drop_last=True)

    validset = Pclass(X_valid, y_valid)
    valid_loader = DataLoader(validset, batch_size=test_batch_size, shuffle=False, num_workers=8, drop_last=True)  



    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = MultiLayerFCNet(input_size, hidden_size, output_size)
    model = nn.DataParallel(model)
    model.to(device)
    #model.load_state_dict(torch.load('path'))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    BestACC=0
    best_loss = float('inf')  # Initialize best loss
    patience_counter = 0  # Initialize patience counter


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

        print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {running_loss/len(train_loader):.4f}')

        model.eval()
        correct = 0
        total = 0
        val_loss=0
        with torch.no_grad():

            for instances, labels in valid_loader:
                instances, labels = instances.to(device), labels.to(device)
                output = model(instances)
                loss = criterion(output, labels)
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            val_loss /= len(valid_loader)
            ACC = 100 * correct / total
            print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {ACC:.2f}%')

            # Check for early stopping based on validation loss
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), '../models/best_model_main.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break
            # print(f'Accuracy: {ACC:.2f}%')
            # if ACC > BestACC:
            #     BestACC = ACC
            #     torch.save(model.state_dict(), 'best_model_main.pth')
                # torch.save(model.state_dict())
                # torch.save(model.state_dict(), 'path')
        model.train()




    # Evaluate on test set using the model with best validation loss
    model.load_state_dict(torch.load('../models/best_model_main.pth'))
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
    test_accuracy = 100 * test_correct / test_total
    print(f'Test Accuracy: {test_accuracy:.2f}%')
