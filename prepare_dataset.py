import torch
import numpy as np
import pandas as pd
import random
import os
import gc
from data_cleaner import DataCleaner
from dataset import BadmintonDataset
from torch.utils.data import DataLoader

def prepare_test_datasets(args):
    matches = DataCleaner(args)
    
    used_column = [
        'rally_id', 'player', 'type',
        'player_location_area', 'opponent_location_area',
        'ball_round', 'set', 'match_id',
        'getpoint_player',
    ]

    matches = matches[used_column]

    type_codes, type_uniques = pd.factorize(matches['type'])
    matches['type'] = type_codes + 1
    args['type_num'] = len(type_uniques) + 1

    data_dir = './data/'

    test_rally_data = pd.read_csv('./data/test.csv')
    test_dataset = BadmintonDataset(test_rally_data, used_column, args)
    test_dataloader = DataLoader(test_dataset, batch_size=args['test_batch_size'], shuffle=False, num_workers=8)
    
    return test_dataloader, data_dir, used_column

def get_fold_dataloader(fold_idx, data_dir, used_column, args):
    """Load data for a single fold."""
    train_rally_data = pd.read_csv(os.path.join(data_dir, f'train_fold_{fold_idx + 1}.csv'))
    valid_rally_data = pd.read_csv(os.path.join(data_dir, f'val_fold_{fold_idx + 1}.csv'))
    # Check player distribution
    print(f"Fold {fold_idx + 1} Train player distribution:\n", train_rally_data['player'].value_counts(normalize=True).sort_index())
    print(f"Fold {fold_idx + 1} Validation player distribution:\n", valid_rally_data['player'].value_counts(normalize=True).sort_index())

    # 移除或填補 NaN/Inf
    train_rally_data = train_rally_data.fillna(0)  # 示例：用 0 填補 NaN
    train_rally_data = train_rally_data.replace([np.inf, -np.inf], 0)
    valid_rally_data = valid_rally_data.fillna(0)
    valid_rally_data = valid_rally_data.replace([np.inf, -np.inf], 0)

    # Create datasets
    train_dataset = BadmintonDataset(train_rally_data, used_column, args)
    valid_dataset = BadmintonDataset(valid_rally_data, used_column, args)

    # Set random seed for reproducibility
    g = torch.Generator()
    g.manual_seed(0)

    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args['train_batch_size'],
        shuffle=True,
        num_workers=8, 
        generator=g
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args['valid_batch_size'],
        shuffle=False,
        num_workers=8,
        generator=g
    )

    # Clear memory
    del train_rally_data, valid_rally_data
    gc.collect()

    return train_dataloader, valid_dataloader