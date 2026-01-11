# 🧬 e2e-ml-pipeline — Gestational Diabetes Prediction Pipeline

This repository implements a reproducible end-to-end machine learning pipeline using DVC for data and pipeline versioning, and MLflow (via DagsHub) for experiment tracking and model registry. The project trains and evaluates a Random Forest classifier on a healthcare dataset to predict diabetes outcomes from clinical indicators.

## 🌍 Overview

This project demonstrates how to build a production-style ML workflow with:

-Automated pipeline orchestration (DVC stages)

-Data and model versioning

-Reproducible execution across environments

-Experiment tracking and artifact logging (MLflow)

### <b> Pipeline stages:</b>

1. Preprocess raw data into a consistent processed dataset

2. Train a model with hyperparameter tuning and MLflow logging

3. Evaluate the trained model and log metrics/artifacts

## 🩺 Problem Context: Gestational Diabetes (GDM)

Gestational Diabetes Mellitus (GDM) is a form of diabetes that can develop during pregnancy due to insulin resistance and increased metabolic demand.

<b> Why tracking and treatment are important? </b>

Early detection and treatment of gestational diabetes is essential to reduce risk of:

-Pregnancy and delivery complications

-Adverse neonatal outcomes

-Increased likelihood of developing type 2 diabetes later in life

### <b>How machine learning can help?</b>

Machine learning can assist healthcare teams by enabling:

early risk identification from structured clinical measurements

consistent screening decision support

improved prioritization of patients who require additional testing and monitoring

### 📊 Dataset

The dataset is tabular and includes clinical features such as:

Pregnancies

Glucose

BloodPressure

SkinThickness

Insulin

BMI

DiabetesPedigreeFunction

Age

Target:

Outcome = 1 → diabetes positive

Outcome = 0 → diabetes negative

Example schema:

<img width="815" height="321" alt="Screenshot 2026-01-11 at 2 59 53 PM" src="https://github.com/user-attachments/assets/1679d8fa-b9e3-4e7a-9a8c-880f61027d03" />

[diabetes.csv dataset (GitHub source)](https://github.com/maghfera/Diabetes-prediction/blob/main/diabetes.csv)


## ✨ Key Features

### DVC Pipeline Orchestration

Structured stages: preprocess → train → evaluate

Reproducible pipeline execution using dvc repro

Dependency-based stage re-execution when code/data/params change

### Data + Model Versioning with DVC

Raw data tracked via .dvc files

Processed datasets and model artifacts handled outside of Git

Optional DVC remote support (DagsHub, S3, etc.)

### Experiment Tracking with MLflow (DagsHub Integration)

Each run logs:

Hyperparameters (from grid search tuning)

Metrics (accuracy, F1, precision/recall for class 1, ROC-AUC, PR-AUC)

### Artifacts:

confusion matrix

classification report

test split snapshot

registered model

### Hyperparameter Optimization

Uses GridSearchCV to tune:

n_estimators

max_depth

min_samples_split

min_samples_leaf

## Results Summary (Stable Run)

Initial experimentation included 50+ runs. Early runs showed instability and overfitting due to inconsistent train/test splits. The pipeline was later corrected using:

fixed random_state

stratified splitting

consistent evaluation using a saved test split

## Tech Stack

<b> Language: Python </b>

<b> Modeling: scikit-learn (RandomForestClassifier) </b>

<b> Pipeline: DVC </b>

<b> Experiment Tracking: MLflow </b>

<b> Platform: DagsHub, Github </b>

<b> Version Control: Git </b>

## 📁 Repository Structure
```bash
e2e-ml-pipeline/
│
├── data/
│ ├── raw/ # raw dataset tracked via DVC
│ └── processed/ # processed data output
│
├── models/
│ └── model.pkl # trained model artifact
│
├── src/
│ ├── preprocess.py
│ ├── train.py
│ └── evaluate.py
│
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── requirements.txt
└── README.md
```
## 🚀 Setup & Usage

1. Create and activate environment
   python -m venv venv
   source venv/bin/activate

2. Install dependencies
   pip install -r requirements.txt

3. Run full pipeline
   dvc repro

4. Push artifacts to DVC remote (optional)
   dvc push

## ⚙️ DVC Stage Creation (Reference)
<b><i> dvc stage add -n preprocess \ </b></i>
 -p preprocess.input,preprocess.output \
 -d src/preprocess.py -d data/raw/data.csv \
 -o data/processed/data.csv \
 python src/preprocess.py

<b><i> dvc stage add -n train \  </b></i>
 -p train.data,train.model,train.random_state,train.n_estimators,train.max_depth \
 -d src/train.py -d data/raw/data.csv \
 -o models/model.pkl \
 python src/train.py

<b><i> dvc stage add -n evaluate \  </b></i>
 -d src/evaluate.py -d models/model.pkl -d data/raw/data.csv \
 python src/evaluate.py

## Data Pipeline Visualization 

<img width="1329" height="623" alt="Screenshot 2026-01-11 at 3 08 11 PM" src="https://github.com/user-attachments/assets/24a09daa-7b30-4cf2-ade0-2fabcc4d4e0a" />


## 🔐 Notes on Credentials

This repository uses a local .env file (ignored via .gitignore) to store MLflow/DagsHub credentials securely.
