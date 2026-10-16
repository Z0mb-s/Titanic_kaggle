import pandas as pd

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

"""print(df_tr.head())"""