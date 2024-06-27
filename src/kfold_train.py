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
from trainAI_main import Pclass, MultiLayerFCNet

if __name__ == '__main__':
    batch_size = 35
    test_batch_size = 35
    input_size = 64 * 64  # 1 channel, 64x64 image size
    hidden_size = 50  # Number of hidden units
    output_size = 4  # Number of output classes (Our data set has 4 classes)
    epochs = 20
    patience = 3

    # get full dataset
    with open('../full_dataset.pkl', 'rb') as f:
        splits = pickle.load(f)
    X = splits['images']
    Y = splits['labels']

    # get folds
    with open('../kfold_dataset.pkl', 'rb') as f:
        folds = pickle.load(f)

    # iterate over the folds
    for i, (train_index, test_index) in enumerate(folds):
        print(f'------------------------------\nModel {i+1}\n------------------------------')
        # get the data
        train_index = np.array(train_index, dtype=int)
        test_index = np.array(test_index, dtype=int)
        x_temp = [X[m] for m in train_index]
        x_test = [X[m] for m in test_index]
        y_temp = [Y[m] for m in train_index]
        y_test = [Y[m] for m in test_index]

        x_train, x_valid, y_train, y_valid = train_test_split(x_temp, y_temp, test_size=0.15, random_state=42, stratify=y_temp)

        trainset = Pclass(x_train, y_train)
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)

        testset = Pclass(x_test, y_test)
        test_loader = DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=8, drop_last=True)

        validset = Pclass(x_valid, y_valid)
        valid_loader = DataLoader(validset, batch_size=test_batch_size, shuffle=False, num_workers=8, drop_last=True)

        # setup model
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = MultiLayerFCNet(input_size, hidden_size, output_size)
        model = nn.DataParallel(model)
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        BestACC = 0
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

            print(f'Epoch [{epoch + 1}/{epochs}], Training Loss: {running_loss / len(train_loader):.4f}')

            model.eval()
            correct = 0
            total = 0
            val_loss = 0
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
                    torch.save(model.state_dict(), f'../kfold_models/model{i}.pth')
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print("Early stopping triggered")
                        break
            model.train()

        # Evaluate on test set using the model with best validation loss
        model.load_state_dict(torch.load(f'../kfold_models/model{i}.pth'))
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
        print(f'Test Accuracy: {test_accuracy:.2f}%\n\n')
