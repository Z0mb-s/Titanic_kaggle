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
        'num_epochs': 50,
        'early_stopping_epochs': 15,
        'lr': 1e-3,
        
        'mixed_precisions': True,
        'gradient_accumulation': False,
        'gradient_clipping': False,
        'gradient_accumulation_steps': 8,
        'clip_value': 2,
        
        'warmup_sheduler': True,
        'warm_epochs': 5,
        'warm_multiplier': 100,
        
        'debug': True,
        'numder_of_train_debug_samples': 5000,
        'numder_of_val_debug_samples': 1000,
        
        'device': 'cuda',
        'save_best': True,
        'save_last': False,
        'verbose_plots': False,
        'resume_from_latest_checkpoint': False,
    },
    'dataloader_params': {
        'batch_size': 32,
        'num_workers': 0,
        'pin_memory': False,
        'presistent_workers': True,
        'shuffle': True,
        'drop_last': True,
    },
    'split': {
        'all_data_train': False,
        'n_splits': 5,
        'folds_to_train': [0, 1, 2, 3, 4],
        'folds_to_submit': [0, 1, 2, 3, 4], 
        'already_split': False,
    },
    'model': {
        'hidden_dim_1': 64,
        'hidden_dim_2': 32,
        'dropout': 0.2,
    },
    'augmentations': {
        'scale': (0.8,1.5),
        'shift': (-0.1,0.1),
        'degree': (-15,15),
        'p': 0.5,
    },
    'optimizer': {
        'name': 'Adam',
        'params': {
            'lr': '${training.lr}',
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
    'loss': {
        'name': '',
        'params': {
            's': 45,
            'm': 0.4,
            'crit': 'bce',
            'class_weights_norm': 'batch',
        },
    },
    'metric': {
        'name': 'accuracy_score',
        'params': {
        },
    },
    'logging': {
        'prints': True,
        'txt_file': True,
        'wandb': False,
        'telegram': False,
        'wandb_username': '...',
        'wandb_project_name': '...',
    },
}

config = OmegaConf.create(config)