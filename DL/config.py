from omegaconf import OmegaConf

config = {
    'general': {
        'experement_name': 'name',
        'seed': 0xDEAD,
        'num_classes': 1,
    },
    'paths': {
        'train_csv': 'data/train.csv',
        'test_csv': 'data/test.csv',
        'submission_csv': 'submission.csv',
        'checkpoints_dir': 'checkpoints/',  
    },
    'training': {
        'num_epochs': 100,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'device': 'cuda',
    },
    'dataloader_params': {
        'batch_size': 32,
        'num_workers': 0,
        'shuffle': True,
    },
    'split': {
        'n_splits': 5,
        'folds_to_train': [0, 1, 2, 3, 4],
        'folds_to_submit': [0, 1, 2, 3, 4],
    },
    'model': {
        'hidden_dim_1': 64,
        'hidden_dim_2': 32,
        'dropout': 0.2,
    },
    'optimizer': {
        'name': 'Adam',
        'params': {
            'lr': '${training.lr}',
            'weight_decay': '${training.weight_decay}',
        },
    },
    'sheduller': {
        'name': 'CosineAnnalingLR',
        'interval': 'epoch',
        'params': {
            'T_max': 195,
            'eta_min': 1e-8,
        },
    },
    'metric': {
        'name': 'accuracy_score',
        'params': {
        },
    },
}

config = OmegaConf.create(config)