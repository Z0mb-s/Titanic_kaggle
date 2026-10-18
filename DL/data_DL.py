import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from config import config


data_tr=pd.read_csv(config.paths.train_csv)
data_ts=pd.read_csv(config.paths.test_csv)

df_tr = data_tr.copy()
df_ts = data_ts.copy()

#transform data
num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
])

prep = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, ['SibSp', 'Pclass', 'Fare', 'Age', 'Parch']),
        ('cat', cat_pipeline, ['Sex', 'Embarked']),
    ]
)

#rename data 
target_col = 'Survived'
drop_cols = ['Id', target_col]

X_tr = df_tr.drop(columns=drop_cols, errors='ignore')
y_tr = df_tr[target_col]
X_ts = df_ts.drop(columns=drop_cols, errors='ignore')


def get_folds(n_splits=config.split.n_splits, seed=config.general.seed):
    """Generate StratifiedKFold splits with leak-free data preprocessing."""

    skf = StratifiedKFold(n_splits=n_splits, shuffle=config.dataloader_params.shuffle, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_tr, y_tr)):
        #raw data
        X_train_raw = X_tr.iloc[train_idx]
        Y_train = y_tr.iloc[train_idx]
        X_val_raw = X_tr.iloc[val_idx]
        Y_val = y_tr.iloc[val_idx]
        X_ts_raw = X_ts
        
        #preprocessing
        X_train_prep = prep.fit_transform(X_train_raw)
        X_val_prep = prep.transform(X_val_raw)
        X_ts_prep = prep.transform(X_ts_raw)
        
        yield fold, X_train_prep, X_val_prep, Y_train, Y_val, X_ts_prep