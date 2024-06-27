# This file is to evaluate the precision, accuracy, recall and f-scores of the kfold models.

import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from torchvision.transforms import transforms
from trainAI_main import MultiLayerFCNet, Pclass  # Import from trainAI_main
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch.nn as nn

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

    y_pred_conf = y_pred.copy()
    y_true_conf = y_true.copy()
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
    return accuracy, report, conf_matrix, y_pred_conf, y_true_conf


if __name__ == '__main__':

    # get full dataset
    with open('../full_dataset.pkl', 'rb') as f:
        splits = pickle.load(f)
    X = splits['images']
    Y = splits['labels']

    # get folds
    with open('../kfold_dataset.pkl', 'rb') as f:
        folds = pickle.load(f)

    # arrays to contain the different evaluations of the 10 models
    accs = []
    reports = []
    matrices = []
    overall_preds = []
    overall_labels = []

    for i, (train_index, test_index) in enumerate(folds):
        print(f'------------------------------\nModel {i+1}\n------------------------------')
        # get the data
        test_index = np.array(test_index, dtype=int)
        x_test = [X[m] for m in test_index]
        y_test = [Y[m] for m in test_index]

        testset = Pclass(x_test, y_test)
        test_loader = DataLoader(testset, batch_size=35, shuffle=False, num_workers=8, drop_last=True)
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        # Load and evaluate models
        main_acc, main_report, main_conf_matrix, y_predict, y_labels = evaluate_model(f'../kfold_models/model{i}.pth', MultiLayerFCNet)
        accs.append(main_acc)
        reports.append(main_report)
        matrices.append(main_conf_matrix)
        overall_preds.extend(y_predict)
        overall_labels.extend(y_labels)

    # make confusion matrix for aggregated kfold models
    overall_preds = np.array(overall_preds)
    overall_labels = np.array(overall_labels)
    cm = confusion_matrix(overall_labels, overall_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['angry', 'engaged', 'happy', 'neutral'])
    disp.plot()
    plt.title('Aggregation of Kfold Models')
    plt.show()

    # # Plot confusion matrices
    # for title, matrix in zip(["Main Model", "Variant 1", "Variant 2"], [main_conf_matrix, variant1_conf_matrix, variant2_conf_matrix]):
    #     disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=['angry', 'engaged', 'happy', 'neutral'])
    #     disp.plot()
    #     plt.title(title)
    #     plt.show()

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
        'Model': ['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5', 'Model 6', 'Model 7', 'Model 8', 'Model 9', 'Model 10'],
        'P_macro': [reports[0]['macro avg']['precision'], reports[1]['macro avg']['precision'], reports[2]['macro avg']['precision'], reports[3]['macro avg']['precision'], reports[4]['macro avg']['precision'], reports[5]['macro avg']['precision'], reports[6]['macro avg']['precision'], reports[7]['macro avg']['precision'], reports[8]['macro avg']['precision'], reports[9]['macro avg']['precision']],
        'R_macro': [reports[0]['macro avg']['recall'], reports[1]['macro avg']['recall'], reports[2]['macro avg']['recall'], reports[3]['macro avg']['recall'], reports[4]['macro avg']['recall'], reports[5]['macro avg']['recall'], reports[6]['macro avg']['recall'], reports[7]['macro avg']['recall'], reports[8]['macro avg']['recall'], reports[9]['macro avg']['recall']],
        'F_macro': [reports[0]['macro avg']['f1-score'], reports[1]['macro avg']['f1-score'], reports[2]['macro avg']['f1-score'], reports[3]['macro avg']['f1-score'], reports[4]['macro avg']['f1-score'], reports[5]['macro avg']['f1-score'], reports[6]['macro avg']['f1-score'], reports[7]['macro avg']['f1-score'], reports[8]['macro avg']['f1-score'], reports[9]['macro avg']['f1-score']],
        'P_micro': [reports[0]['micro avg']['precision'], reports[1]['micro avg']['precision'], reports[2]['micro avg']['precision'], reports[3]['micro avg']['precision'], reports[4]['micro avg']['precision'], reports[5]['micro avg']['precision'], reports[6]['micro avg']['precision'], reports[7]['micro avg']['precision'], reports[8]['micro avg']['precision'], reports[9]['micro avg']['precision']],
        'R_micro': [reports[0]['micro avg']['recall'], reports[1]['micro avg']['recall'], reports[2]['micro avg']['recall'], reports[3]['micro avg']['recall'], reports[4]['micro avg']['recall'], reports[5]['micro avg']['recall'], reports[6]['micro avg']['recall'], reports[7]['micro avg']['recall'], reports[8]['micro avg']['recall'], reports[9]['micro avg']['recall']],
        'F_micro': [reports[0]['micro avg']['f1-score'], reports[1]['micro avg']['f1-score'], reports[2]['micro avg']['f1-score'], reports[3]['micro avg']['f1-score'], reports[4]['micro avg']['f1-score'], reports[5]['micro avg']['f1-score'], reports[6]['micro avg']['f1-score'], reports[7]['micro avg']['f1-score'], reports[8]['micro avg']['f1-score'], reports[9]['micro avg']['f1-score']],
        'Accuracy': [accs[0], accs[1], accs[2], accs[3], accs[4], accs[5], accs[6], accs[7], accs[8], accs[9]]
    }

    df_metrics = pd.DataFrame(metrics)
    print("Metrics summary:")
    print(df_metrics)
