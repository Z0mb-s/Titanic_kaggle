import pandas as pd
import numpy as np
from copy import deepcopy
from datetime import datetime

from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from config_ML import config
from data import get_folds, df_tr, df_ts, y_tr, features, cat_features

model_name = config.general.select_model
model_params = dict(config.model[model_name])
# globals search import class
base_model = globals()[model_name](**model_params) if model_name != 'KNeighborsClassifier' else KNeighborsClassifier()

n_splits=config.general.n_splits
oof_preds=np.zeros(len(df_tr), dtype=int)
test_preds_sum=np.zeros(len(df_ts))
fold_accuracies=[]

#get data splits
for fold, (X_train_prep, X_val_prep, Y_train, Y_val, X_ts_prep, 
           X_train_raw, X_val_raw, X_ts_raw, val_idx) in get_folds(n_splits=n_splits, seed=config.general.seed):
    
    # copy for iter fold
    model = deepcopy(base_model)

    #linearregression check
    if isinstance(model, (LinearRegression, Lasso, Ridge, ElasticNet)):
        model.fit(X_train_prep, Y_train)
        val_pred=(model.predict(X_val_prep) >= 0.5).astype(int).flatten()
        test_pred=(model.predict(X_ts_prep) >= 0.5).astype(int).flatten()
    #knn check
    elif isinstance(model, KNeighborsClassifier):
        param_grid=dict(config.model.KNeighborsClassifier)
        grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
        grid.fit(X_train_prep, Y_train)
        #выбираем лучшую модель
        model = grid.best_estimator_
        val_pred = model.predict(X_val_prep).flatten()
        test_pred = model.predict(X_ts_prep).flatten()
    #catboost check
    elif isinstance(model, CatBoostClassifier):
        model.fit(X_train_raw, Y_train, cat_features=cat_features)
        val_pred = model.predict(X_val_raw).astype(int).flatten()
        test_pred = model.predict(X_ts_raw).astype(int).flatten()
    elif isinstance(model, LGBMClassifier):
        model.fit(X_train_prep, Y_train)
        val_pred=model.predict(X_val_prep).astype(int).flatten()
        test_pred=model.predict(X_ts_prep).astype(int).flatten()

    else:
        model.fit(X_train_prep, Y_train)
        val_pred = model.predict(X_val_prep).astype(int).flatten()
        test_pred = model.predict(X_ts_prep).astype(int).flatten()

    # fold ansvers
    oof_preds[val_idx] = val_pred
    test_preds_sum += test_pred
    
    fold_acc = accuracy_score(Y_val, val_pred)
    fold_accuracies.append(fold_acc)
    print(f"Fold {fold + 1}/{n_splits} | Val Accuracy: {fold_acc:.4f}")
    
acc = accuracy_score(y_tr, oof_preds)
# mean ansvers from folds
test_predictions = (test_preds_sum >= (n_splits / 2)).astype(int)

print(f'\nOOF predicts sum: {oof_preds.sum()}')
print(f'Mean Fold Accuracy: {np.mean(fold_accuracies):.4f}')
print(f'Total OOF Accuracy: {acc:.4f}')

#save model metrics
model_name = base_model.__class__.__name__
current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
with open('results.txt', 'a', encoding='utf-8') as f:
    f.write('=' * 60 + '\n')
    f.write(f'Timestamp:       {current_time}\n')
    f.write(f'Experiment:      {config.general.experement_name}\n')
    f.write(f'Model Class:     {model_name}\n')
    f.write(f'Validation:      StratifiedKFold ({n_splits} splits)\n')
    f.write(f'Input Features:  {features}\n')
    f.write(f'Random Seed:     {config.general.seed}\n')
    f.write(f'Fold Accuracies: {[round(x, 4) for x in fold_accuracies]}\n')
    f.write(f'OOF Accuracy:    {acc:.4f}\n')
    f.write(f'Model Params:    {base_model.get_params()}\n')
    f.write('=' * 60 + '\n\n')

print(f'Результаты эксперимента [{model_name}] успешно добавлены в results.txt')

submission = pd.DataFrame({
    'PassengerId': df_ts['PassengerId'],
    'Survived': test_predictions
})
submission.to_csv('submission.csv', index=False)
print("Файл submission.csv готов к отправке!")