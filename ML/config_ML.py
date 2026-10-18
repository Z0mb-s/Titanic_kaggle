from omegaconf import OmegaConf

config = {
    'general': {
        'experement_name': 'titanic_1',
        'seed': 0xDEAD,
        'class': 'classifier',
        'n_splits': 5,
        'shuffle': True,
        'select_model': 'CatBoostClassifier',
    },
    'paths': {
        'train_csv': 'data/train.csv',
        'test_csv': 'data/test.csv',
        'submission_csv': 'submission.csv',  
    },
    'model': {
        'LogisticRegression': {},
        'LinearRegression': {},
        'RandomForestClassifier': {
            'n_estimators': 100, 
            'max_depth': 9, 
            'random_state': '${general.seed}'
            },
        'DecisionTreeClassifier': {},
        'KNeighborsClassifier': {
            'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
            'weights': ['uniform', 'distance'],
            'metric': ['minkowski'],
            'p': [1, 2],
            },
        'CatBoostClassifier': {
            'iterations': 300,
            'learning_rate': 5e-2,
            'depth': 6,
            'random_seed': '${general.seed}',
            'verbose': False,
            },
        'LGBMClassifier': {
            'learning_rate': 3e-2,
            'num_leaves': 9,
            'n_estimators': 100,
            'random_state': '${general.seed}',
            'verbose': -1,
            },
        'Lasso': {
            'alpha': 0.01,
            'random_state': '${general.seed}',
        },
        'Ridge': {
            'alpha': 1.0,
            'random_state': '${general.seed}',
        },
        'ElasticNet': {
            'alpha': 0.1,
            'l1_ratio': 0.5,
            'random_state': '${general.seed}',
        }
    }

}
config = OmegaConf.create(config)