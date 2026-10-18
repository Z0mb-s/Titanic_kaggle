import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from config_ML import config

data_tr=pd.read_csv(config.paths.train_csv)
data_ts=pd.read_csv(config.paths.test_csv)

df_tr = data_tr.copy()
df_ts = data_ts.copy()

#rename data 
target_col = 'Survived'
num_features = ['SibSp', 'Pclass', 'Fare', 'Age', 'Parch']
cat_features = ['Sex', 'Embarked']
features = num_features + cat_features 

X_tr = df_tr[features].copy()
y_tr = df_tr[target_col].copy()
X_ts = df_ts[features].copy()

X_tr['Embarked']=X_tr['Embarked'].fillna('missing').astype(str)
X_ts['Embarked']=X_ts['Embarked'].fillna('missing').astype(str)

X_tr['Sex']=X_tr['Sex'].astype(str)
X_ts['Sex']=X_ts['Sex'].astype(str)

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


# folds generator
def get_folds(n_splits=config.general.n_splits, seed=config.general.seed):
    """Generate StratifiedKFold splits with leak-free data preprocessing."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=config.general.shuffle, random_state=seed)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_tr, y_tr)):
        # raw data from indexes
        X_train_raw = X_tr.iloc[train_idx]
        Y_train = y_tr.iloc[train_idx]
        X_val_raw = X_tr.iloc[val_idx]
        Y_val = y_tr.iloc[val_idx]
        X_ts_raw = X_ts
        
        # preprocessing
        X_train_prep = prep.fit_transform(X_train_raw)
        X_val_prep = prep.transform(X_val_raw)
        X_ts_prep = prep.transform(X_ts_raw)
        # for boosts
        X_train_raw[cat_features] = X_train_raw[cat_features].astype(str)
        X_val_raw[cat_features] = X_val_raw[cat_features].astype(str)
        X_ts_raw[cat_features] = X_ts_raw[cat_features].astype(str)
        
        yield fold, (
            X_train_prep, X_val_prep, Y_train, Y_val, X_ts_prep,
            X_train_raw, X_val_raw, X_ts_raw, val_idx
        )