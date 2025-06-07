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
    
    train_index = []
    valid_index = []
    test_index = []

    for match_id in matches['match_id'].unique():
        match = matches[matches['match_id']==match_id]
        rally_index = match['rally_id'].unique()
        #np.random.shuffle(rally_index) 
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
    
