# Kaggle Titanic - ML pipeline

## 1.0

A hands-on Machine Learning project built around Kaggle Titanic: Machine Learning from Disaster competition. The goal of this repository is to practice building a clean, reproductable ML pipeline - from exploratory data analysis and feature preprocessing to hyperparameter tuning and experiment tracking.

## 1.1

In project i realise moduled pipeline for data analyse, data preparation, preprocessing models, training various machine learning models, and logging experements results.

### 2.0 I use:

1. I use various models such as: LogisticRegression, LinearRegression, Lasso, Ridge, ElasticNet, RandomForestClassifier, DecisionTreeClassifier, KNeighborsClassifier
2. In requirements.txt you can install libraries what i use. (use pip install -r requirements.txt)
3. In results.txt, you can see some the logged experiment history, including model hyperparameters, validation accuracy, and predictions for each run.

### 2.1 Imports

- **Python 3.10+**
- **Pandas, NumPy** — data manipulation and preparation
- **Scikit-Learn** — preprocessing (`ColumnTransformer`, `StandardScaler`, `OneHotEncoder`), validation (`GridSearchCV`) and base models
- **CatBoost** — gradient boosting
- **OmegaConf** — experement configuration management

### 2.2 Project Structure

#### Structure

├── data/
│   ├── train.csv               # Training dataset
│   ├── test.csv                # Test dataset
│   └── gender_submission.csv   # Kaggle baseline benchmark
├── config_ML.py                # Hyperparameter configuration (OmegaConf)
├── data.py                     # Data loading and missing value imputation
├── data_research.py            # Exploratory Data Analysis (EDA)
├── main.py                     # Main training and validation pipeline
├── results.txt                 # Experiment logs and evaluation metrics
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation

### 3.1 Models & Experimental Results

All experiments were evaluated using a hold-out validation set (20%) and 5-fold cross-validation (5-fold CV):

- **LogisticRegression / LinearRegression** — baseline accuracy ~0.78[cite: 6]
- **KNeighborsClassifier (GridSearchCV)** — hyperparameter tuning (`n_neighbors`, `weights`, `p`)[cite: 6]
- **RandomForestClassifier / DecisionTreeClassifier** — accuracy ~0.80–0.83[cite: 6]
- **CatBoostClassifier** — gradient boosting with native categorical feature support

All experimental runs, parameters, and evaluation metrics are automatically logged to `results.txt`[cite: 2, 6].