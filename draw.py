import json
import numpy as np
import matplotlib.pyplot as plt
import os

shot_type_labels = {1: 'short service', 2: 'clear', 3: 'push & rush', 4: 'smash', 5: 'defensive return', 
                  6: 'drive', 7: 'net shot', 8: 'lob', 9: 'drop', 10: 'long service'}
location_labels = {
    1: "Front-Right", 2: "Front-Center", 3: "Back-Left", 4: "Middle-Left",
    5: "Back-Center", 6: "Middle-Center", 7: "Net Zone Center", 8: "Middle-Right",
    9: "Back-Right", 10: "Out"
}

# 載入 rally_analysis.json
output_folder = './'
with open(os.path.join(output_folder, 'rally_analysis.json'), 'r') as f:
    rally_analysis = json.load(f)

# 選擇一個拉力進行分析（例如第一個拉力）
rally_idx = 0
rally_data = rally_analysis[rally_idx]

# 提取數據
b2a_contrib = np.array(rally_data['A_weight'])  # (67, 64)
a2b_contrib = np.array(rally_data['B_weight'])  # (67, 64)
shot_type = np.array(rally_data['shot_type'])  # (67,)
player_A_loc = np.array(rally_data['player_A_loc'])  # (67,)
player_B_loc = np.array(rally_data['player_B_loc'])  # (67,)
hit_area = np.array(rally_data['hit_area'])  # (67,)
win_prob = rally_data['win_prob']

# 檢查是否有 NaN 或 Inf
if np.any(np.isnan(b2a_contrib)) or np.any(np.isinf(b2a_contrib)):
    print(f"Warning: NaN or Inf in b2a_contrib for rally {rally_idx}")
if np.any(np.isnan(a2b_contrib)) or np.any(np.isinf(a2b_contrib)):
    print(f"Warning: NaN or Inf in a2b_contrib for rally {rally_idx}")

# 計算有效序列長度（假設 shot_type 的非零部分表示有效時間步）
valid_length = np.sum(shot_type != 0)
print(f"Valid sequence length: {valid_length}")

# 僅保留有效時間步
b2a_contrib = b2a_contrib[:valid_length, :]
a2b_contrib = a2b_contrib[:valid_length, :]
shot_type = shot_type[:valid_length]
player_A_loc = player_A_loc[:valid_length]
player_B_loc = player_B_loc[:valid_length]
hit_area = hit_area[:valid_length]

# 計算 B 對 A 和 A 對 B 的貢獻範數
b2a_contrib_norm = np.linalg.norm(b2a_contrib, axis=-1)  # (valid_length,)
a2b_contrib_norm = np.linalg.norm(a2b_contrib, axis=-1)  # (valid_length,)

# 可視化 B 對 A 和 A 對 B 的貢獻範數曲線
plt.figure(figsize=(12, 6))
plt.plot(range(valid_length), b2a_contrib_norm, label='B to A Contribution Norm', marker='o')
plt.plot(range(valid_length), a2b_contrib_norm, label='A to B Contribution Norm', marker='s')
plt.xlabel('Time Steps')
plt.ylabel('Contribution Norm')
plt.title(f'Contribution Strength (Rally {rally_idx}, Win Prob: {win_prob:.3f})')
plt.legend()

# 添加擊球類型和位置標註
for t in range(valid_length):
    plt.text(t, b2a_contrib_norm[t], 
             f"{shot_type_labels.get(int(shot_type[t]), 'Unknown')} ({location_labels.get(int(player_B_loc[t]), 'Unknown')})",
             fontsize=8, ha='center', va='bottom', color='red')
    plt.text(t, a2b_contrib_norm[t], 
             f"{shot_type_labels.get(int(shot_type[t]), 'Unknown')} ({location_labels.get(int(player_A_loc[t]), 'Unknown')})",
             fontsize=8, ha='center', va='top', color='blue')

plt.tight_layout()
plt.savefig(os.path.join(output_folder, f'contrib_norm_rally_{rally_idx}.png'))
plt.show()

# 找出 B 對 A 的最大影響時間步
top_k = 3
b2a_max_indices = np.argsort(b2a_contrib_norm)[-top_k:][::-1]  # B 對 A 影響最大的時間步
print(f"Top {top_k} time steps where B has the largest impact on A:")
for idx in b2a_max_indices:
    print(f"Time step {idx}: B's shot {shot_type_labels.get(int(shot_type[idx]), 'Unknown')} "
          f"at {location_labels.get(int(player_B_loc[idx]), 'Unknown')} "
          f"impacts A's shot {shot_type_labels.get(int(shot_type[idx]), 'Unknown')} "
          f"at {location_labels.get(int(player_A_loc[idx]), 'Unknown')} "
          f"(Contribution Norm: {b2a_contrib_norm[idx]:.3f})")

# 找出 A 對 B 的最大影響時間步
a2b_max_indices = np.argsort(a2b_contrib_norm)[-top_k:][::-1]  # A 對 B 影響最大的時間步
print(f"\nTop {top_k} time steps where A has the largest impact on B:")
for idx in a2b_max_indices:
    print(f"Time step {idx}: A's shot {shot_type_labels.get(int(shot_type[idx]), 'Unknown')} "
          f"at {location_labels.get(int(player_A_loc[idx]), 'Unknown')} "
          f"impacts B's shot {shot_type_labels.get(int(shot_type[idx]), 'Unknown')} "
          f"at {location_labels.get(int(player_B_loc[idx]), 'Unknown')} "
          f"(Contribution Norm: {a2b_contrib_norm[idx]:.3f})")