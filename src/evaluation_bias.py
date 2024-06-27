import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from torchvision.transforms import transforms
from trainAI_main import MultiLayerFCNet  # Import from trainAI_main
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import os
import torch.nn as nn

# Define dataset class
class Pclass(Dataset):
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transforms.Compose([
            transforms.Resize(size=(64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = self.transform(Image.open(self.image_paths[idx]).convert('L'))
        label = self.labels[idx]
        return image, label

# Function to load images and labels from directories
def load_images_from_folder(folder_path):
    image_paths = []
    labels = []
    for label, class_name in enumerate(['angry', 'engaged', 'happy', 'neutral']):
        class_folder = os.path.join(folder_path, class_name)
        for filename in os.listdir(class_folder):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_paths.append(os.path.join(class_folder, filename))
                labels.append(label)
    return image_paths, labels

# Function to evaluate model
def evaluate_model(model_path, test_loader):
    model = MultiLayerFCNet(64 * 64, 50, 4)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    y_pred = []
    y_true = []

    with torch.no_grad():
        for instances, labels in test_loader:
            instances, labels = instances.to(device), labels.to(device)
            output = model(instances)
            _, predicted = torch.max(output.data, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Calculate macro-averaged metrics
    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    report['macro avg'] = {
        'precision': macro_precision,
        'recall': macro_recall,
        'f1-score': macro_f1
    }

    return accuracy, report, conf_matrix

# Main block
if __name__ == '__main__':
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_path = input("Type the model name you would like to evaluate on the bias attributes (i.e. 'best_model_main.pth'): ")
    model_path = '../models/' + model_path

    # Evaluate the main model on the bias attribute dataset
    age_groups = ['Young', 'Middle-aged', 'Senior']
    gender_groups = ['Male', 'Female', 'Other/Non-binary']
    metrics = {
        'Age': {group: {'#Images': 0, 'Accuracy': 0, 'Precision': 0, 'Recall': 0, 'F1-Score': 0} for group in age_groups + ['Total/Average']},
        'Gender': {group: {'#Images': 0, 'Accuracy': 0, 'Precision': 0, 'Recall': 0, 'F1-Score': 0} for group in gender_groups + ['Total/Average']}
    }

    # Define the paths for the age attribute dataset
    age_paths = {
        'Young': '../dataset-bias_attributes/age/young',
        'Middle-aged': '../dataset-bias_attributes/age/middle-aged',
        'Senior': '../dataset-bias_attributes/age/senior'
    }

    total_age_images = 0
    avg_accuracy_age = 0
    avg_precision_age = 0
    avg_recall_age = 0
    avg_f1score_age = 0

    # Evaluate for each age group
    for age_group, folder_path in age_paths.items():
        image_paths, labels = load_images_from_folder(folder_path)
        testset = Pclass(image_paths, labels)
        test_loader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=8, drop_last=True)

        accuracy, report, conf_matrix = evaluate_model(model_path, test_loader)

        metrics['Age'][age_group]['#Images'] = len(image_paths)
        metrics['Age'][age_group]['Accuracy'] = accuracy
        metrics['Age'][age_group]['Precision'] = report['macro avg']['precision']
        metrics['Age'][age_group]['Recall'] = report['macro avg']['recall']
        metrics['Age'][age_group]['F1-Score'] = report['macro avg']['f1-score']
        
        total_age_images = total_age_images + len(image_paths)
        avg_accuracy_age = avg_accuracy_age + accuracy
        avg_precision_age = avg_precision_age + report['macro avg']['precision']
        avg_recall_age = avg_recall_age + report['macro avg']['recall']
        avg_f1score_age = avg_f1score_age + report['macro avg']['f1-score']
    
    metrics['Age']['Total/Average'] = {'#Images': total_age_images, 'Accuracy': avg_accuracy_age/3, 'Precision': avg_precision_age/3, 'Recall': avg_recall_age/3, 'F1-Score': avg_f1score_age/3}


    # Define the paths for the gender attribute dataset
    gender_paths = {
        'Male': '../dataset-bias_attributes/gender/male',
        'Female': '../dataset-bias_attributes/gender/female',
        'Other/Non-binary': '../dataset-bias_attributes/gender/other'
    }

    total_gender_images = 0
    avg_accuracy_gender = 0
    avg_precision_gender = 0
    avg_recall_gender = 0
    avg_f1score_gender = 0

    # Evaluate for each gender group
    for gender_group, folder_path in gender_paths.items():
        image_paths, labels = load_images_from_folder(folder_path)
        testset = Pclass(image_paths, labels)
        test_loader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=8, drop_last=True)

        accuracy, report, conf_matrix = evaluate_model(model_path, test_loader)

        metrics['Gender'][gender_group]['#Images'] = len(image_paths)
        metrics['Gender'][gender_group]['Accuracy'] = accuracy
        metrics['Gender'][gender_group]['Precision'] = report['macro avg']['precision']
        metrics['Gender'][gender_group]['Recall'] = report['macro avg']['recall']
        metrics['Gender'][gender_group]['F1-Score'] = report['macro avg']['f1-score']
        
        total_gender_images = total_gender_images + len(image_paths)
        avg_accuracy_gender = avg_accuracy_gender + accuracy
        avg_precision_gender = avg_precision_gender + report['macro avg']['precision']
        avg_recall_gender = avg_recall_gender + report['macro avg']['recall']
        avg_f1score_gender = avg_f1score_gender + report['macro avg']['f1-score']

    metrics['Gender']['Total/Average'] = {'#Images': total_gender_images, 'Accuracy': avg_accuracy_gender/3, 'Precision': avg_precision_gender/3, 'Recall': avg_recall_gender/3, 'F1-Score': avg_f1score_gender/3}

    # Fill in the table
    table = pd.DataFrame(columns=['Attribute', 'Group', '#Images', 'Accuracy', 'Precision', 'Recall', 'F1-Score'],
                         dtype='object')
    for attribute in metrics:
        for group in metrics[attribute]:
            new_row = pd.DataFrame({
                'Attribute': [attribute],
                'Group': [group],
                '#Images': [metrics[attribute][group]['#Images']],
                'Accuracy': [metrics[attribute][group]['Accuracy']],
                'Precision': [metrics[attribute][group]['Precision']],
                'Recall': [metrics[attribute][group]['Recall']],
                'F1-Score': [metrics[attribute][group]['F1-Score']]
            })
            table = pd.concat([table, new_row], ignore_index=True)

    # Calculate overall averages
    overall_avg = {
        'Attribute': 'Overall System',
        'Group': 'Total/Average',
        '#Images': total_age_images + total_gender_images,
        'Accuracy': (avg_accuracy_age + avg_accuracy_gender)/6,
        'Precision': (avg_precision_age + avg_precision_gender)/6,
        'Recall': (avg_recall_age + avg_recall_gender)/6,
        'F1-Score': (avg_f1score_age + avg_f1score_gender)/6
    }

    # Add the overall average row to the table
    table = pd.concat([table, pd.DataFrame([overall_avg])], ignore_index=True)

    print(table)