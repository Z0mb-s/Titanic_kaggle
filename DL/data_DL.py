import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import config


data_tr=pd.read_csv('data/train.csv')
data_ts=pd.read_csv('data/test.csv')

df_tr = data_tr.copy()
df_ts = data_ts.copy()

df_tr['Age']=df_tr['Age'].fillna(df_tr['Age'].mean())
df_ts['Age']=df_ts['Age'].fillna(df_ts['Age'].mean())

df_tr['Embarked']=df_tr['Embarked'].fillna(df_tr['Embarked'].mode()[0])
df_ts['Embarked']=df_ts['Embarked'].fillna(df_ts['Embarked'].mode()[0])

df_tr['Pclass']=df_tr['Pclass'].fillna(df_tr['Pclass'].mode()[0])
df_ts['Pclass']=df_ts['Pclass'].fillna(df_ts['Pclass'].mode()[0])

df_tr['Fare'] = df_tr['Fare'].fillna(df_tr['Fare'].median())
df_ts['Fare'] = df_ts['Fare'].fillna(df_ts['Fare'].median())

#features and data
features=['SibSp', 'Pclass', 'Fare','Age', 'Parch', 'Embarked', 'Sex']
X_tr=df_tr[features]
y_tr=df_tr['Survived']
X_ts=df_ts[features]

#Transform data
prep=ColumnTransformer(
    transformers=[
        (
            'cat',
            OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'),
            ['Sex', 'Embarked'],
        ),
        (
            'num',
            StandardScaler(),
            ['SibSp', 'Pclass', 'Fare', 'Age', 'Parch'],
        ),
    ]
)

#split data
def get_data_splits(seed=config.general.seed, test_size=0.2):
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_tr, y_tr, test_size=test_size, random_state=seed
    )
    X_train_prep = prep.fit_transform(X_train)
    X_val_prep = prep.transform(X_val)
    X_ts_prep = prep.transform(X_ts)
    
    return X_train_prep, X_val_prep, Y_train, Y_val, X_ts_prep