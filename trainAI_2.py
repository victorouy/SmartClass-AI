import torch
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


class Pclass(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)    
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

    def load_dataset(root_path):
        image_paths = []
        labels = []
        class_names = ['angry', 'engaged', 'happy', 'neutral']
        for idx, class_name in enumerate(class_names):
            class_dir = os.path.join(root_path, class_name)
            for img_name in os.listdir(class_dir):
                image_paths.append(os.path.join(class_dir, img_name))
                labels.append(idx)
        return image_paths, labels

    # Load the dataset
    train_paths, train_labels = load_dataset('dataset-cleaned/train')
    test_paths, test_labels = load_dataset('dataset-cleaned/test')

    # Split train data into train and validation sets
    train_paths, val_paths, train_labels, val_labels = train_test_split(train_paths, train_labels, test_size=0.2, stratify=train_labels, random_state=42)

    # Transform
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Create Dataset objects
    train_dataset = Pclass(train_paths, train_labels, transform=transform)
    val_dataset = Pclass(val_paths, val_labels, transform=transform)
    test_dataset = Pclass(test_paths, test_labels, transform=transform)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=8, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8, drop_last=True)


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
        val_correct = 0
        val_total = 0
        with torch.no_grad():

            for instances, labels in val_loader:
                instances, labels = instances.to(device), labels.to(device)
                output = model(instances)
                _, predicted = torch.max(output.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_accuracy = 100 * val_correct / val_total
        print(f'Validation Accuracy: {val_accuracy:.2f}%')
        if val_accuracy > BestACC:
            BestACC = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
        model.train()

        print(f'Best Validation Accuracy: {BestACC:.2f}%')

    # Evaluate on test set
    model.load_state_dict(torch.load('best_model_main.pth'))
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





