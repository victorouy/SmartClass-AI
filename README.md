# COMP-472-Project

Github URL: https://github.com/jungsoolee1/COMP-472-Project

## Team AK_18

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

### Steps to execute code for model training (Main model + variants)

1. From the root folder, enter `cd src`.
2. Enter `python split_dataset.py` to split dataset for which the models will train.
3. Executing scripts for training models:
   a. Main Model: enter `python trainAI_main.py`
   b. Variant 1: enter `python variant1.py`
   c. Variant 2: enter `python variant2.py`


### Steps to load the model and test it on data

1. From the root folder, enter `cd src`.
2. Enter `python evaluation.py` to load the model.
3. Type "Dataset" if you wanna test the whole dataset or "single" if you wanna predict a single image.
4. If you typed "single", type the category of the image (angry, engaged, happy, or neutral).
5. If you typed "single", then type the full filepath of the image you wish to predict.
