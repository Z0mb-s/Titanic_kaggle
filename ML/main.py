import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from catboost import CatBoost, CatBoostClassifier
#int bostings in this code, and do dl pipeline!

from datetime import datetime

from data import df_tr, df_ts
from config_ML import config

#rename data 
features=['SibSp', 'Pclass', 'Fare','Age', 'Parch', 'Embarked', 'Sex']
X_tr=df_tr[features]
y_tr=df_tr['Survived']
X_ts=df_ts[features]
model = CatBoostClassifier(**dict(config.model.CatBoostClassifier))

#transform data
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
X_train, X_test, Y_train, Y_test = train_test_split(
    X_tr, y_tr, test_size=0.2, random_state=config.general.seed
    )

#featuring?
X_train_prep = prep.fit_transform(X_train)
X_test_prep = prep.transform(X_test)
X_ts_prep = prep.transform(X_ts)

#training and check model type
model.fit(X_train_prep, Y_train)
#linearregression check
if isinstance(model, (LinearRegression, Lasso, Ridge, ElasticNet)):
    model.fit(X_train_prep, Y_train)
    Y_pred_cont=model.predict(X_test_prep)
    Y_pred=(Y_pred_cont >= 0.5).astype(int)
    test_predictions_cont=model.predict(X_ts_prep)
    test_predictions=(test_predictions_cont >= 0.5).astype(int)
#knn check
elif isinstance(model, KNeighborsClassifier):
    param_grid=dict(config.model.KNeighborsClassifier)
    grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid.fit(X_train_prep, Y_train)
    print(f"Лучшие параметры: {grid.best_params_}")
    print(f"Лучший CV score:  {grid.best_score_:.4f}")
    #выбираем лучшую модель
    model = grid.best_estimator_
    Y_pred = model.predict(X_test_prep)
    test_predictions = model.predict(X_ts_prep)
#catboost check
elif isinstance(model, CatBoostClassifier):
    features = ['Sex', 'Embarked']
    model.fit(X_train, Y_train, cat_features=features)
    Y_pred = model.predict(X_test).astype(int)
    test_predictions = model.predict(X_ts).astype(int)

else:
    model.fit(X_train_prep, Y_train)
    Y_pred = model.predict(X_test_prep).astype(int)
    test_predictions = model.predict(X_ts_prep).astype(int)

acc = accuracy_score(Y_test, Y_pred)

print(f'predicts: {Y_pred.sum()}')
print(f'accuracy: {acc:.2f}')

#save model metrics
model_name = model.__class__.__name__
current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
with open('results.txt', 'a', encoding='utf-8') as f:
    f.write('=' * 60 + '\n')
    f.write(f'Timestamp:       {current_time}\n')
    f.write(f'Experiment:      {config.general.experement_name}\n')
    f.write(f'Model Class:     {model_name}\n')
    f.write(f'Input Features:  {features}\n')
    f.write(f'Random Seed:     {config.general.seed}\n')
    f.write(f'Accuracy:        {acc:.4f}\n')
    f.write(f'Model Params:    {model.get_params()}\n')
    f.write('-' * 60 + '\n')
    f.write(f'Real labels (test):\n{list(Y_test.values)}\n')
    f.write(f'Predictions (test):\n{list(Y_pred)}\n')
    f.write('=' * 60 + '\n\n')

print(f'Результаты эксперимента [{model_name}] успешно добавлены в results.txt')

print("Результаты успешно записаны в results.txt")

submission = pd.DataFrame({
    'PassengerId': df_ts['PassengerId'],
    'Survived': test_predictions
})
submission.to_csv('submission.csv', index=False)
print("Файл submission.csv готов к отправке!")