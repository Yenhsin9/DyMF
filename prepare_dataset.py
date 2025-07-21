import torch
import numpy as np
import pandas as pd
import random
import os
from data_cleaner import DataCleaner
from dataset import BadmintonDataset
from torch.utils.data import DataLoader
from torch.utils.data import default_collate
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

def dynamic_collate(batch):
    # batch: list of (rally, target) tuples
    rallies, label_tensors = zip(*batch)     # ← 這一步就把 batch 拆成 rallies 與 labels
    B = len(rallies)

    # 把 labels 堆起來
    labels = torch.stack(label_tensors).long()

    # 找出最長序列
    Lmax = rallies[0][0].shape[0] 

    # 準備 padded tensors
    rally_batch = {
        'player':    torch.zeros(B, Lmax, dtype=torch.long),
        'shot_type': torch.zeros(B, Lmax, dtype=torch.long),
        'A_x':       torch.zeros(B, Lmax),
        'A_y':       torch.zeros(B, Lmax),
        'B_x':       torch.zeros(B, Lmax),
        'B_y':       torch.zeros(B, Lmax),
    }

    # 單一迴dlly 解構、填值
    for i, rally in enumerate(rallies):
        player, shot_type, A_x, A_y, B_x, B_y, seq_len = rally
        rally_batch['player'][i]    = torch.from_numpy(player)
        rally_batch['shot_type'][i] = torch.from_numpy(shot_type)
        rally_batch['A_x'][i]       = torch.from_numpy(A_x)
        rally_batch['A_y'][i]       = torch.from_numpy(A_y)
        rally_batch['B_x'][i]       = torch.from_numpy(B_x)
        rally_batch['B_y'][i]       = torch.from_numpy(B_y)

    return rally_batch, labels


def prepare_test_datasets(args):
    matches = DataCleaner(args)
    
    used_column = [
        'rally_id', 'player', 'type',
        'player_location_area', 'opponent_location_area',
        'hit_area','backhand', 'aroundhead',
        'player_location_x', 'player_location_y',
        'opponent_location_x', 'opponent_location_y',
        'ball_round', 'set', 'match_id',
        'getpoint_player', 'roundscore_A', 'roundscore_B',
        'consecutive_points','score_diff'
    ]

    matches = matches[used_column]

    type_codes, type_uniques = pd.factorize(matches['type'])
    matches['type'] = type_codes + 1
    args['type_num'] = len(type_uniques) + 1

    data_dir = './data/'

    test_rally_data = pd.read_csv('./data/test.csv')
    test_dataset = BadmintonDataset(test_rally_data, used_column, args)
    test_dataloader = DataLoader(test_dataset, batch_size=args['test_batch_size'], shuffle=False, num_workers=8)

    return test_dataloader
    

def prepare_kfold_datasets(args, k_folds=1):
    matches = DataCleaner(args)
    
    used_column = [
        'rally_id', 'player', 'type',
        'player_location_area', 'opponent_location_area',
        'hit_area','backhand', 'aroundhead',
        'player_location_x', 'player_location_y',
        'opponent_location_x', 'opponent_location_y',
        'ball_round', 'set', 'match_id',
        'getpoint_player', 'roundscore_A', 'roundscore_B',
        'consecutive_points','score_diff'
    ]

    matches = matches[used_column]

    type_codes, type_uniques = pd.factorize(matches['type'])
    matches['type'] = type_codes + 1
    args['type_num'] = len(type_uniques) + 1

    data_dir = './data/'

    test_rally_data = pd.read_csv('./data/test.csv')
    test_dataset = BadmintonDataset(test_rally_data, used_column, args)
    test_dataloader = DataLoader(test_dataset, batch_size=args['test_batch_size'], shuffle=False, num_workers=8)
    
    fold_datasets = []
    for i in range(1, k_folds+1):
        train_rally_data = pd.read_csv(os.path.join(data_dir, f'train_fold_{i}.csv'))
        valid_rally_data = pd.read_csv(os.path.join(data_dir, f'val_fold_{i}.csv'))
        # 检查 player_id 分布
        print(f"Fold {i} Train player distribution:\n", train_rally_data['player'].value_counts(normalize=True).sort_index())
        print(f"Fold {i} Validation player distribution:\n", valid_rally_data['player'].value_counts(normalize=True).sort_index())

        # 创建数据集
        train_dataset = BadmintonDataset(train_rally_data, used_column, args)
        valid_dataset = BadmintonDataset(valid_rally_data, used_column, args)

        # 设置随机种子以确保可重复性
        g = torch.Generator()
        g.manual_seed(0)

        # 创建数据加载器
        train_dataloader = DataLoader(train_dataset, batch_size=args['train_batch_size'], shuffle=True, num_workers=8, generator=g)
        valid_dataloader = DataLoader(valid_dataset, batch_size=args['valid_batch_size'], shuffle=False, num_workers=8, generator=g)

        fold_datasets.append((train_dataloader, valid_dataloader, args))

    return fold_datasets,test_dataloader


