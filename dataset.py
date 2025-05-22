from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
from DyMF.model import initialize_adjacency_matrix
PAD = 0


class BadmintonDataset(Dataset):
    def __init__(self, data, used_column, args):
        super(BadmintonDataset).__init__()
        self.player_sequence = []
        self.court_sequence = []
        self.target_sequence = []
        self.adj=[]

        rally_id_grouped = data.groupby('rally_id').groups
        rally_data = data[used_column]    
        self.encode_length=0
  
        for rally_id in rally_id_grouped.values():
            rally_id = rally_id.to_numpy()
            one_rally = rally_data.iloc[rally_id].reset_index(drop=True)
            
            seqence_length = len(one_rally)   
            if seqence_length > self.encode_length:
                self.encode_length=seqence_length    # maxlength in dataset

        pre_setid = None
        consecutive_points = 0
        pre_diff=None
        pre_getPoint=None
        for rally_id in rally_id_grouped.values():
            rally_id = rally_id.to_numpy()
            one_rally = rally_data.iloc[rally_id].reset_index(drop=True)

            seqence_length = len(one_rally)     #shot sequence length
            tmpGetPoint = one_rally['getpoint_player'].iloc[-1]

            # rally information
            player = one_rally[['player']].values.reshape(-1)
            shot_type = one_rally[['type']].values.reshape(-1)
            scoreA = one_rally[['roundscore_A']].values.reshape(-1)[0]
            scoreB = one_rally[['roundscore_B']].values.reshape(-1)[0]
            tmp = scoreA-scoreB
            player_x = one_rally[['player_location_x']].values.reshape(-1)
            player_y = one_rally[['player_location_y']].values.reshape(-1)
            opponent_x = one_rally[['opponent_location_x']].values.reshape(-1)
            opponent_y = one_rally[['opponent_location_y']].values.reshape(-1)

            setid = one_rally['set'].iloc[-1]
             # (Consecutive Points)
            if setid != pre_setid: 
                roundscore_diff = 0
                consecutive_points = 0
            else:
                roundscore_diff = pre_diff
                if pre_getPoint == 'A':
                    consecutive_points += 1
                else:
                    consecutive_points = 0 
            pre_getPoint = tmpGetPoint
            pre_diff = tmp
            pre_setid = setid
       
            player = np.pad(player, (0, self.encode_length - seqence_length), 'constant', constant_values=(0))
            shot_type  = np.pad(shot_type , (0, self.encode_length - seqence_length), 'constant', constant_values=(0))
            player_x = np.pad(player_x, (0, self.encode_length - seqence_length), 'constant', constant_values=(0))
            player_y = np.pad(player_y, (0, self.encode_length - seqence_length), 'constant', constant_values=(0))
            opponent_x = np.pad(opponent_x, (0, self.encode_length - seqence_length), 'constant', constant_values=(0))
            opponent_y = np.pad(opponent_y, (0, self.encode_length - seqence_length), 'constant', constant_values=(0))      

            mask = np.concatenate([np.ones(seqence_length, dtype=np.float32),
                                   np.zeros(self.encode_length - seqence_length, dtype=np.float32)])
            
            player_A_x = np.empty((self.encode_length,), dtype=float)
            player_A_x[0::2] = player_x[0::2]
            player_A_x[1::2] = opponent_x[1::2]
            player_A_y = np.empty((self.encode_length,), dtype=float)
            player_A_y[0::2] = player_y[0::2]
            player_A_y[1::2] = opponent_y[1::2]

            player_B_x = np.empty((self.encode_length,), dtype=float)
            player_B_x[0::2] = opponent_x[0::2]
            player_B_x[1::2] = player_x[1::2]
            player_B_y = np.empty((self.encode_length,), dtype=float)
            player_B_y[0::2] = opponent_y[0::2]
            player_B_y[1::2] = player_y[1::2]
            #print(rally_id,roundscore_diff,consecutive_points)
            self.player_sequence.append([player, shot_type, player_A_x, player_A_y, player_B_x, player_B_y, seqence_length,mask,roundscore_diff,consecutive_points])
          
            # predict target
            label = 1 if tmpGetPoint == 'A' else 0
            self.target_sequence.append(label)
 
    def __len__(self):
        return len(self.player_sequence)
    
    def __getitem__(self, index):
        rally = self.player_sequence[index]  
        target = torch.tensor(self.target_sequence[index], dtype=torch.float32)
        return rally,target