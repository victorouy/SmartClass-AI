# Artificial Intelligence: Deep Learning Project

The objective of this project is to develop a Deep Learning Convolutional Neural Network (CNN)
using PyTorch that can analyze images of students in a classroom or online meeting setting and
categorize them into distinct states or activities. The 4 classes of facial expressions our articial neural network can predict are Neutral, Engaged, Angry, and Happy.

## Team members

Data Specialist: Nadim Khalife

Training Specialist: Jungsoo Lee

Evaluation Specialist: Victor-Thyreth Ouy

## Content of the submission

<ins>src/data_cleaning.py</ins> : Python script for data cleaning of our dataset

<ins>src/visualization.py</ins>: Python script to create the data visualization

<ins>dataset_information.md</ins>: Document detailing the sources and licensing of our dataset (https://www.kaggle.com/datasets/msambare/fer2013)

<ins>AI Project Part 1 - Report.pdf</ins>: Project Report

<ins>Originality Form folder</ins>: Signed Expectation of Originality forms from each team member

<ins>dataset folder</ins>: Folder containing the dataset images

<ins>dataset-cleaned folder</ins>: Folder created after running data_cleaning.py that contains the cleaned dataset images

## How To Run Python Scripts

Note: The following steps are assuming you are running our code with the same folder hierarchy as shown in our Github repository.

### First step to executing ANY code. You must create a virtual environment using pip

1. Make sure you are in the root folder when typing commands in the terminal.
2. Enter `py -m venv .venv` for Windows or `python3 -m venv .venv` for Unix/macOS.
3. Enter `.venv\Scripts\activate` for Windows or `source .venv/bin/activate` for unix/macOS to activate the environment.
4. Enter `pip install -r requirements.txt` to install packages.

### Steps to execute code for data cleaning + labeling

1. From the root folder, enter `cd src`.
2. Enter `python data_cleaning.py` to execute the script.
3. This script will create or replace a folder in the root dir called 'dataset-cleaned'

### Steps to execute code for data visualization

1. From the root folder, enter `cd src`.
2. Enter `python visualization.py` to execute the script.

### Steps to execute code for model training (main model + variants)

1. From the root folder, enter `cd src`.
2. Enter `python split_dataset.py` to split dataset for which the models will train.
   * a. Enter the relative path of the dataset you would like to split and train off of.
3. Executing scripts for training models:
   * a. Main Model: enter `python trainAI_main.py`
   * b. Variant 1: enter `python variant1.py`
   * c. Variant 2: enter `python variant2.py`

### Steps to evaluate models

1. You first need to train the models (in steps above)
2. From the root folder, enter `cd src`.
3. Enter `python evaluation_models.py`

### Steps to load the model and test it on data

1. From the root folder, enter `cd src`.
2. Enter `python evaluation.py` to load the model.
3. Type "Dataset" if you wanna test the whole dataset or "single" if you wanna predict a single image.
4. If you typed "single", type the category of the image (angry, engaged, happy, or neutral).
5. If you typed "single", then type the full filepath of the image you wish to predict
   c. Variant 2: enter `python variant2.py`

### Steps to evaluate model on bias attributes (age & gender)

1. From the root folder, enter `cd src`.
2. You first need to split the training data.
   * a. Enter `python split_dataset.py` to split dataset.
   * b. Then you need to enter the relative path of the dataset (either one of the following):
      - Level 1: enter `../dataset-bias_level1/`
      - Level 2: enter `../dataset-bias_level2/`
      - Level 3: enter `../dataset-bias_level3/`
3. Create and train the bias models (either one of the following):
   * a. Level 1: enter `python trainAI_bias1.py`
   * b. Level 2: enter `python trainAI_bias2.py`
   * c. Level 3: enter `python trainAI_bias3.py`
4. Now, enter `python evaluation_bias.py` to evaluate the model based on the biases.
   * a. Enter the model name you would like to evaluate
      - Level 1: enter `model_bias1.pth`
      - Level 2: enter `model_bias2.pth`
      - Level 3: enter `model_bias3.pth`


### Steps to training with k-fold cross-validation

1. From the root folder, enter `cd src`.
2. You first need to obtain the folds for the k-fold cross-validation.
   * a. Enter `python kfolding.py` to obtain and save the folds.
3. Then, train with k-fold cross-validation by entering `python kfold_train.py`.

### Steps to evaluate the k-fold models.

1. From the root folder, enter `cd src`.
2. Enter `python evaluation_kfold.py` to obtain the performance metrics and confusion matrix.
