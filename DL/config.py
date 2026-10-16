from omegaconf import OmegaConf

config = {
    'general': {
        'experement_name': 'name',
        'seed': 0xDEAD,
        'num_classes': 250,
    },
    'paths': {
        'path_to_csv': 'path....',
        'path_to_data': 'path....',
        'path_to_labels': 'path....',
        'path_to_json': 'path....',
        'path_to_folder': 'path....',
        'pq_path': 'path....',
        'path_to_checkpoints': 'path....${general.experement_name}',
    },
    'training': {
        'num_epochs': 200,
        'early_stopping_epochs': 100,
        'lr': 1e-4 / 100,
        
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
        'batch_size': 64,
        'num_workers': 8,
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
        'freeze_batchnorms': False,
        'converter_sample_input_shape': [111, 222, 1 ],
        'model_sample_input_shape': [111, 333],
        'params': {
            'max_length': 111,
            'embed_dim': 222,
            'num_point': 333,
            'num_head': 16,
            'num_class': '${general.num_classes}',
            'mu,_block': 1,
        },
    },
    'augmentations': {
        'scale': (0.8,1.5),
        'shift': (-0.1,0.1),
        'degree': (-15,15),
        'p': 0.5,
    },
    'optimizer': {
        'name': '/custom/ranger',
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
        'name': '/custom/ArcFaceLoss',
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
        'wandb': True,
        'telegram': False,
        'wandb_username': '...',
        'wandb_project_name': '...',
    },
}

config = OmegaConf.create(config)