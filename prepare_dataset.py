import torch
import numpy as np
import pandas as pd
import random
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
        seq_len = int(seq_len)
        print(seq_len)
        rally_batch['player'][i]    = torch.from_numpy(player)
        rally_batch['shot_type'][i] = torch.from_numpy(shot_type)
        rally_batch['A_x'][i]       = torch.from_numpy(A_x)
        rally_batch['A_y'][i]       = torch.from_numpy(A_y)
        rally_batch['B_x'][i]       = torch.from_numpy(B_x)
        rally_batch['B_y'][i]       = torch.from_numpy(B_y)

    return rally_batch, labels


def prepare_dataset(args):
    matches = DataCleaner(args)
    
    used_column = [
    'rally_id','player','type',
    'player_location_area','opponent_location_area',
    'hit_area',
    'player_location_x','player_location_y',
    'opponent_location_x','opponent_location_y',
    'ball_round','set','match_id',
    'getpoint_player','roundscore_A','roundscore_B'    
    ]

    matches = matches[used_column]

    player_codes, player_uniques = pd.factorize(matches['player'])
    matches['player'] = player_codes + 1
    args['player_num'] = len(player_uniques) + 1

    type_codes, type_uniques = pd.factorize(matches['type'])
    matches['type'] = type_codes + 1
    args['type_num'] = len(type_uniques) + 1
    
    rally_ids = matches['rally_id'].unique()

    # 隨機分割測試集
    np.random.seed(args['seed'])
    np.random.shuffle(rally_ids)
    test_num = int(len(rally_ids) * args['test_ratio'])
    test_rally_ids = rally_ids[:test_num]
    train_val_rally_ids = rally_ids[test_num:]

    test_rally_data = matches[matches['rally_id'].isin(test_rally_ids)].reset_index(drop=True)
    test_dataset = BadmintonDataset(test_rally_data, used_column, args)
    test_dataloader = DataLoader(test_dataset, batch_size=args['test_batch_size'], shuffle=False, num_workers=8)

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=args['seed'])
    fold_datasets = []

    for match_id in matches['match_id'].unique():
        match = matches[matches['match_id']==match_id]
        rally_index = match['rally_id'].unique()
       # np.random.shuffle(rally_index) 
        train_num = int(len(rally_index) * args['train_ratio'])
        valid_num = int(len(rally_index) * args['valid_ratio'])

        train_index.extend(rally_index[:train_num])
        valid_index.extend(rally_index[train_num:train_num+valid_num])
        test_index.extend(rally_index[train_num+valid_num:])
    
    train_index = np.array(train_index)
    valid_index = np.array(valid_index)
    test_index = np.array(test_index)
 
    train_rally_data = matches[matches['rally_id'].isin(train_index)].reset_index(drop=True)
    valid_rally_data = matches[matches['rally_id'].isin(valid_index)].reset_index(drop=True)
    test_rally_data = matches[matches['rally_id'].isin(test_index)].reset_index(drop=True)

    train_dataset = BadmintonDataset(train_rally_data, used_column, args)
    valid_dataset = BadmintonDataset(valid_rally_data, used_column, args)
    test_dataset = BadmintonDataset(test_rally_data, used_column, args)

    g = torch.Generator()
    g.manual_seed(0)

    train_dataloader = DataLoader(train_dataset, batch_size=args['train_batch_size'], shuffle=True, num_workers=8)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args['valid_batch_size'], shuffle=False, num_workers=8)
    test_dataloader = DataLoader(test_dataset, batch_size=args['test_batch_size'], shuffle=False, num_workers=8)
    return train_dataloader, valid_dataloader, test_dataloader, args
    


def prepare_kfold_datasets(args, k_folds=1):
    matches = DataCleaner(args)
    
    used_column = [
    'rally_id','player','type',
    'player_location_area','opponent_location_area',
    'hit_area',
    'player_location_x','player_location_y',
    'opponent_location_x','opponent_location_y',
    'ball_round','set','match_id',
    'getpoint_player','roundscore_A','roundscore_B'    
    ]

    matches = matches[used_column]

    player_codes, player_uniques = pd.factorize(matches['player'])
    matches['player'] = player_codes + 1
    args['player_num'] = len(player_uniques) + 1

    type_codes, type_uniques = pd.factorize(matches['type'])
    matches['type'] = type_codes + 1
    args['type_num'] = len(type_uniques) + 1
    
    rally_ids = matches['rally_id'].unique()
    rally_player_map = matches[['rally_id', 'player']].drop_duplicates(subset=['rally_id'])
    print(f"Number of rows in rally_player_map before deduplication: {len(rally_player_map)}")  # Debug

    # 檢查 rally_player_map 是否與 rally_ids 長度一致
    if len(rally_player_map) != len(rally_ids):
        raise ValueError(f"Mismatch: rally_player_map has {len(rally_player_map)} rows, but rally_ids has {len(rally_ids)}")
    
    rally_player_map = rally_player_map.set_index('rally_id')
    
    # 檢查 stratify 陣列長度
    stratify_values = rally_player_map.loc[rally_ids, 'player']
    print(f"Length of stratify values: {len(stratify_values)}")  # Debug: should be 2372

    # 分層分割 rally_id
    train_val_rally_ids, test_rally_ids = train_test_split(
        rally_ids,
        test_size=args['test_ratio'],
        stratify=stratify_values,
        random_state=args['seed']
    )
    
    # 檢查資料洩漏
    assert len(np.intersect1d(test_rally_ids, train_val_rally_ids)) == 0, "Overlap detected between test_rally_ids and train_val_rally_ids!"

    # 提取測試集資料
    test_rally_data = matches[matches['rally_id'].isin(test_rally_ids)].reset_index(drop=True)
    test_dataset = BadmintonDataset(test_rally_data, used_column, args)
    test_dataloader = DataLoader(test_dataset, batch_size=args['test_batch_size'], shuffle=False, num_workers=8)

    matches = matches[matches['rally_id'].isin(train_val_rally_ids)].copy()
    rally_ids = matches['rally_id'].unique()
  
    rally_player_map2 = matches[['rally_id', 'player']].drop_duplicates(subset='rally_id')
    rally_player_map2 = rally_player_map2.set_index('rally_id')
    
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=args['seed'])
    fold_datasets = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(rally_player_map2.index, rally_player_map2.player)):
        train_idx = rally_player_map2.index[train_idx]
        val_idx = rally_player_map2.index[val_idx]
    
        train_rally_data = matches[matches['rally_id'].isin(train_idx)].reset_index(drop=True)
        valid_rally_data = matches[matches['rally_id'].isin(val_idx)].reset_index(drop=True)
        # 檢查資料洩漏
        assert len(np.intersect1d(train_idx, val_idx)) == 0, "Overlap detected between train_idx and val_idx!"
        
        # 檢查 player_id 分佈
        print("Test player distribution:\n", test_rally_data['player'].value_counts(normalize=True).sort_index())
        print("Train player distribution:\n", train_rally_data['player'].value_counts(normalize=True).sort_index())
        print("val player distribution:\n", valid_rally_data['player'].value_counts(normalize=True).sort_index())

        train_dataset = BadmintonDataset(train_rally_data, used_column, args)
        valid_dataset = BadmintonDataset(valid_rally_data, used_column, args)

        g = torch.Generator()
        g.manual_seed(0)

        train_dataloader = DataLoader(train_dataset, batch_size=args['train_batch_size'], shuffle=True, num_workers=8, generator=g)
        valid_dataloader = DataLoader(valid_dataset, batch_size=args['valid_batch_size'], shuffle=False, num_workers=8, generator=g)

        fold_datasets.append((train_dataloader, valid_dataloader, args))

    return fold_datasets, test_dataloader