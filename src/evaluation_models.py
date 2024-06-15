import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from torchvision.transforms import transforms
from trainAI_main import MultiLayerFCNet  # Import from trainAI_main
from variant1 import MultiLayerFCNet as Variant1Net  # Import from variant1
from variant2 import MultiLayerFCNet as Variant2Net  # Import from variant2
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch.nn as nn

# Define dataset class
class Pclass(Dataset):
    def __init__(self, X, y):
        self.allaimges = X
        self.clsLabel = y
        self.mytransform = transforms.Compose([
            transforms.Resize(size=(64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.allaimges)

    def __getitem__(self, idx):
        Im = self.mytransform(Image.open(self.allaimges[idx]).convert('L'))
        Cls = self.clsLabel[idx]
        return Im, Cls

# Evaluation function
def evaluate_model(model_path, model_class):
    model = model_class(64 * 64, 50, 4)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    y_pred = []
    y_true = []

    with torch.no_grad():
        for idx, (instances, labels) in enumerate(test_loader):
            instances, labels = instances.to(device), labels.to(device)
            output = model(instances)
            _, predicted = torch.max(output.data, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Calculate macro-averaged metrics
    macro_precision = precision_score(y_true, y_pred, average='macro')
    macro_recall = recall_score(y_true, y_pred, average='macro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    report['macro avg'] = {
        'precision': macro_precision,
        'recall': macro_recall,
        'f1-score': macro_f1
    }

    # Calculate micro-averaged metrics
    micro_precision = precision_score(y_true, y_pred, average='micro')
    micro_recall = recall_score(y_true, y_pred, average='micro')
    micro_f1 = f1_score(y_true, y_pred, average='micro')

    report['micro avg'] = {
        'precision': micro_precision,
        'recall': micro_recall,
        'f1-score': micro_f1
    }
    return accuracy, report, conf_matrix

# Main block
if __name__ == '__main__':
    # Load dataset splits
    with open('../dataset_splits.pkl', 'rb') as f:
        splits = pickle.load(f)

    X_test = splits['X_test']
    y_test = splits['y_test']

    testset = Pclass(X_test, y_test)
    test_loader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=8, drop_last=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load and evaluate models
    main_acc, main_report, main_conf_matrix = evaluate_model('../models/best_model_main.pth', MultiLayerFCNet)
    variant1_acc, variant1_report, variant1_conf_matrix = evaluate_model('../models/best_model_variant1.pth', Variant1Net)
    variant2_acc, variant2_report, variant2_conf_matrix = evaluate_model('../models/best_model_variant2.pth', Variant2Net)

    # Plot confusion matrices
    for title, matrix in zip(["Main Model", "Variant 1", "Variant 2"], [main_conf_matrix, variant1_conf_matrix, variant2_conf_matrix]):
        disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=['angry', 'engaged', 'happy', 'neutral'])
        disp.plot()
        plt.title(title)
        plt.show()

    # Summarize metrics in a table
    def extract_metrics(report):
        return {
            'P_macro': report['macro avg']['precision'],
            'R_macro': report['macro avg']['recall'],
            'F_macro': report['macro avg']['f1-score'],
            'P_micro': report['micro avg']['precision'],
            'R_micro': report['micro avg']['recall'],
            'F_micro': report['micro avg']['f1-score'],
        }

    metrics = {
        'Model': ['Main Model', 'Variant 1', 'Variant 2'],
        'P_macro': [main_report['macro avg']['precision'], variant1_report['macro avg']['precision'], variant2_report['macro avg']['precision']],
        'R_macro': [main_report['macro avg']['recall'], variant1_report['macro avg']['recall'], variant2_report['macro avg']['recall']],
        'F_macro': [main_report['macro avg']['f1-score'], variant1_report['macro avg']['f1-score'], variant2_report['macro avg']['f1-score']],
        'P_micro': [main_report['micro avg']['precision'], variant1_report['micro avg']['precision'], variant2_report['micro avg']['precision']],
        'R_micro': [main_report['micro avg']['recall'], variant1_report['micro avg']['recall'], variant2_report['micro avg']['recall']],
        'F_micro': [main_report['micro avg']['f1-score'], variant1_report['micro avg']['f1-score'], variant2_report['micro avg']['f1-score']],
        'Accuracy': [main_acc, variant1_acc, variant2_acc]
    }

    df_metrics = pd.DataFrame(metrics)
    print("Metrics summary:")
    print(df_metrics)
