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

The project is split into two dedicated directories: `ML/` for classical machine learning workflows and `DL/` for PyTorch neural network experiments:

```text
├── data/
│   ├── train.csv               # Training dataset
│   ├── test.csv                # Test dataset
│   └── gender_submission.csv   # Kaggle baseline benchmark
├── ML/
│   ├── config_ML.py            # ML hyperparameter and pipeline configuration (OmegaConf)
│   ├── data.py                 # Data loading, pipelines, and preprocessing
│   ├── data_research.py        # Exploratory Data Analysis (EDA) and visualizations
│   ├── main.py                 # Classical ML training, evaluation, and inference script
│   └── results.txt             # Experiment history and logged evaluation metrics
├── DL/
│   ├── config.py               # DL configuration (epochs, optimizer, lr, layers)
│   ├── data_DL.py              # PyTorch tabular data preprocessing and split generator
│   ├── train_functions.py      # TabularDataset class, train_one_epoch, and eval_one_epoch
│   ├── main_DL.py              # PyTorch MLP training, validation, and inference script
│   └── checkpoints/            # Saved PyTorch model checkpoints (.pth)
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
- Select the active model in `ML/config_ML.py` via `select_model` (e.g., `CatBoostClassifier`, `RandomForestClassifier`, `LGBMClassifier`, `LogisticRegression`).
- Evaluation metrics, random seeds, and hyperparameters are automatically logged to `ML/results.txt`.
- A submission file (`submission.csv`) is automatically generated in the **project root directory**, formatted with `PassengerId` and `Survived` columns ready for direct submission to Kaggle.

### 4.2 Deep Learning Pipeline (PyTorch MLP)
Run the neural network pipeline from the project root:
```bash
python DL/main_DL.py
```
- Network hyperparameters (hidden dimensions, dropout, learning rate, optimizer) are managed in `DL/config.py`.
- The best performing model checkpoint is saved to `DL/checkpoints/`.
- Test set predictions are computed and saved to `submission.csv` in the **project root directory**.

---

## 5. Models & Experimental Results

> **Evaluation Methodology:** All current results are evaluated on a **single hold-out train/validation split**. Full cross-validation results will be reported once multi-fold aggregation is finalized across all modules.

| Model | Category | Validation Accuracy (Single Split) | Highlights & Notes |
|---|---|:---:|---|
| **LogisticRegression / LinearRegression** | Linear Baseline | ~0.78 | Baseline benchmark on scaled numerical & one-hot categorical features |
| **KNeighborsClassifier** | Instance-based | ~0.79 | Tuned via `GridSearchCV` (`n_neighbors`, `weights`, `p`) |
| **RandomForestClassifier / DecisionTree** | Tree Ensembles | ~0.80 – 0.83 | Non-linear feature interactions with controlled tree depth |
| **LightGBM (LGBMClassifier)** | Gradient Boosting | ~0.81 – 0.83 | Fast histogram-based gradient boosting on preprocessed numeric features |
| **CatBoostClassifier** | Gradient Boosting | ~0.82 – 0.84 | Gradient boosting with native handling of categorical features (`Sex`, `Embarked`) |
| **PyTorch MLP (TitanicModel)** | Deep Learning | ~0.79 – 0.81 | Multi-layer Perceptron (Linear -> BatchNorm -> ReLU -> Dropout) trained with Adam/BCEWithLogitsLoss |

All experimental runs, parameters, and evaluation metrics for classical models are tracked in `ML/results.txt`.