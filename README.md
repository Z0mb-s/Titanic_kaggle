# Kaggle Titanic - ML & DL Pipeline

## 1. Overview

A hands-on Machine Learning and Deep Learning project built around the **Kaggle Titanic: Machine Learning from Disaster** competition. The goal of this repository is to establish a clean, reproducible, and modular pipeline — from exploratory data analysis and feature preprocessing to hyperparameter tuning, neural network training, and experiment tracking.

The project implements modular pipelines for data analysis, preprocessing, training various classical and deep learning models, and logging experimental results.

---

## 2. Environment & Requirements

- **Python Version**: Tested and verified on **Python 3.13** *(pinned versions of NumPy and SciPy require Python >= 3.12, making Python 3.10 incompatible)*.
- **Dependencies**: All required packages are listed in `requirements.txt`. Install them via:
  ```bash
  pip install -r requirements.txt
  ```

### Key Libraries
- **Pandas, NumPy** — data manipulation, cleaning, and preparation
- **Scikit-Learn** — data preprocessing (`ColumnTransformer`, `StandardScaler`, `OneHotEncoder`, `SimpleImputer`), hyperparameter search (`GridSearchCV`), and baseline models
- **CatBoost & LightGBM** — gradient boosting algorithms
- **PyTorch** — deep learning framework for the tabular MLP neural network
- **OmegaConf** — flexible hierarchical configuration management (`config_ML.py`, `config.py`)

---

## 3. Project Structure

The project is divided into two dedicated directories: `ML/` for classical machine learning workflows and `DL/` for PyTorch neural network experiments. Execution artifacts (evaluation logs, model checkpoints, submission files) are stored at the root directory of the repository:

```text
├── data/
│   ├── train.csv               # Training dataset
│   ├── test.csv                # Test dataset
│   └── gender_submission.csv   # Kaggle baseline benchmark
├── ML/
│   ├── config_ML.py            # ML hyperparameter and pipeline configuration (OmegaConf)
│   ├── data.py                 # Data loading, pipelines, and StratifiedKFold splits
│   ├── data_research.py        # Exploratory Data Analysis (EDA) and visualizations
│   └── main.py                 # Classical ML training, 5-fold CV, and inference script
├── DL/
│   ├── config.py               # DL configuration (epochs, optimizer, lr, layers)
│   ├── data_DL.py              # PyTorch tabular data preprocessing and split generator
│   ├── train_functions.py      # TabularDataset class, train_one_epoch, and eval_one_epoch
│   └── main_DL.py              # PyTorch MLP training, validation, and inference script
├── checkpoints/                # Saved PyTorch model checkpoints (.pth)
├── results.txt                 # Experiment history and logged evaluation metrics
├── requirements.txt            # Project dependencies
├── submission.csv              # Generated final predictions ready for Kaggle submission
└── README.md                   # Project documentation
```

---

## 4. How to Run

### 4.1 Classical ML Pipeline
Run the classical ML training pipeline from the project root:
```bash
python ML/main.py
```
- Select the active model in `ML/config_ML.py` via `select_model` (e.g., `CatBoostClassifier`, `RandomForestClassifier`, `LGBMClassifier`, `LogisticRegression`, `KNeighborsClassifier`).
- Validation is conducted using **5-Fold Stratified Cross-Validation**.
- Per-fold metrics, overall Out-Of-Fold (OOF) accuracy, random seeds, and hyperparameters are automatically logged to the root-level `results.txt`.
- The final submission file (`submission.csv`), aggregated via majority voting across folds, is generated in the **project root directory**.

### 4.2 Deep Learning Pipeline (PyTorch MLP)
Run the neural network pipeline from the project root:
```bash
python DL/main_DL.py
```
- Hyperparameters (hidden layer sizes, dropout, learning rate, optimizer) are managed in `DL/config.py`.
- Training also utilizes stratified cross-validation (`StratifiedKFold`).
- The best performing model checkpoints are saved directly to the root-level `checkpoints/` directory.
- Test set predictions are averaged across folds and saved to `submission.csv` in the **project root directory**.

---

## 5. Models & Experimental Results

The project validation scheme was updated from an initial single train/validation split to a full 5-fold cross-validation setup with leak-free preprocessing. To maintain an accurate comparison, results are separated according to the validation protocol used.

### 5.1 Current Evaluation: 5-Fold Stratified Cross-Validation (OOF Accuracy)
The current implementation evaluates all models using Out-Of-Fold (OOF) predictions across 5 stratified folds:

| Model | Category | OOF Accuracy (5-Fold CV) | Mean Fold Accuracy | Notes |
|---|---|:---:|:---:|---|
| **CatBoostClassifier** | Gradient Boosting | ~0.83 – 0.84 | ~0.835 | Native handling of categorical features (`Sex`, `Embarked`) |
| **LightGBM (LGBMClassifier)** | Gradient Boosting | ~0.82 – 0.83 | ~0.824 | Histogram-based boosting on numerical and OHE categorical features |
| **RandomForestClassifier** | Tree Ensemble | ~0.81 – 0.83 | ~0.820 | Controlled tree depth (`max_depth=9`) |
| **KNeighborsClassifier** | Instance-based | ~0.79 – 0.81 | ~0.802 | Tuned via `GridSearchCV` with preprocessing isolated inside the pipeline |
| **LogisticRegression** | Linear Baseline | ~0.79 – 0.80 | ~0.796 | Standard linear classifier on scaled numerical features |
| **PyTorch MLP (TitanicModel)** | Deep Learning | ~0.80 – 0.81 | ~0.805 | Linear -> BatchNorm -> ReLU -> Dropout architecture aggregated over 5 folds |

### 5.2 Legacy Baseline: Single Hold-Out Split
*Early experimental benchmarks evaluated on a single 80/20 train/validation split:*

| Model | Category | Validation Accuracy (Single Split) | Notes |
|---|---|:---:|---|
| **Linear / Logistic Baseline** | Linear Baseline | ~0.78 | Initial baseline on preprocessed numerical and OHE features |
| **KNeighborsClassifier** | Instance-based | ~0.79 | Preliminary hyperparameter search prior to pipeline isolation |
| **DecisionTree / Random Forest** | Tree Models | ~0.80 – 0.82 | Initial tree depth tuning |

Detailed logs for every run, individual fold metrics, and model parameters are recorded in `results.txt`.