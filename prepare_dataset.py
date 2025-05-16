import torch
import numpy as np
import pandas as pd
import random
from data_cleaner import DataCleaner
from dataset import BadmintonDataset
from torch.utils.data import DataLoader
from torch.utils.data import default_collate
from DyMF.model import initialize_adjacency_matrix

def dynamic_collate(batch):
    # batch: list of dict from BadmintonDataset.__getitem__
    B = len(batch)
    Ls = [item['length'] for item in batch]
    Lmax = max(Ls)
    Nmax = Lmax * 2  # node 數量
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)

    # 建立儲存 tensor
    out = {
        'player':    torch.zeros(B, Lmax, dtype=torch.long),
        'shot_type': torch.zeros(B, Lmax, dtype=torch.long),
        'A_x':       torch.zeros(B, Lmax),
        'A_y':       torch.zeros(B, Lmax),
        'B_x':       torch.zeros(B, Lmax),
        'B_y':       torch.zeros(B, Lmax),
        'mask':      torch.zeros(B, Lmax),           # 1 表示真實，0 表示 padding
        #'adj':       torch.zeros(B, 13, Nmax, Nmax),  # multi-relational graph
        'label':     labels,
        'encode_length': torch.tensor(Lmax), 
    }
    
    for i, item in enumerate(batch):
        L = item['length']
        # copy 原始序列
        out['player'][i, :L]    = torch.from_numpy(item['player'])
        out['shot_type'][i, :L] = torch.from_numpy(item['shot_type'])
        out['A_x'][i, :L]       = torch.from_numpy(item['A_x'])
        out['A_y'][i, :L]       = torch.from_numpy(item['A_y'])
        out['B_x'][i, :L]       = torch.from_numpy(item['B_x'])
        out['B_y'][i, :L]       = torch.from_numpy(item['B_y'])
        out['mask'][i, :L]      = 1
        
    # shot_type_tensor   = out['shot_type']  # LongTensor[B, Lmax]
    # adjacency_batch    = initialize_adjacency_matrix(B, Lmax, shot_type_tensor) 
    # out['adj'] = adjacency_batch

    return out

def prepare_dataset(args):
    matches = DataCleaner(args)
    
    used_column = [
    'rally_id','player','type',
    'player_location_x','player_location_y',
    'opponent_location_x','opponent_location_y',
    'ball_round','set','match_id',
    'getpoint_player'    
    ]

    matches = matches[used_column]

    player_codes, player_uniques = pd.factorize(matches['player'])
    matches['player'] = player_codes + 1
    args['player_num'] = len(player_uniques) + 1

    type_codes, type_uniques = pd.factorize(matches['type'])
    matches['type'] = type_codes + 1
    args['type_num'] = len(type_uniques) + 1
    
    train_index = []
    valid_index = []
    test_index = []

    for match_id in matches['match_id'].unique():
        match = matches[matches['match_id']==match_id]
        rally_index = match['rally_id'].unique()

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
    
