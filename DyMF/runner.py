import os
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
import torch.distributions.multivariate_normal as torchdist
from torch.nn import BCEWithLogitsLoss
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score
from torch.nn import BCELoss
from DyMF.draw_plot import draw_plot
import pandas as pd
import pickle
import json
from prepare_dataset import get_fold_dataloader
import gc
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

try:
    import seaborn as sns
    import matplotlib.pyplot as plt
    seaborn_available = True
except ImportError:
    seaborn_available = False
    print("Seaborn not installed. Skipping heatmap visualization.")

def train_kfold(data_dir, used_column, k_folds, test_dataloader, encoder, location_criterion, shot_type_criterion, encoder_optimizer, args, device="cpu"):
    bce_loss = BCEWithLogitsLoss()
    patience = args.get('patience', 5)
    encoder.to(device)  # Ensure model is on correct device

    # Store metrics for all folds
    fold_results = {
        'train_auc_list': [],
        'val_auc_list': [],
        'train_brier_list': [],
        'val_brier_list': [],
        'train_loss_list': [],
        'val_loss_list': [],
        'train_acc_list': [],
        'val_acc_list': [],
    }
    fold_val_losses = []
    fold_val_aucs = []
    fold_val_briers = []
    fold_val_acc = []
    best_val_loss = float('inf')
    best_fold = 0
    
    for fold in range(k_folds):
        print(f"\nTraining Fold {fold + 1}/{k_folds}")
        
        # Load data for current fold only
        train_dataloader, valid_dataloader = get_fold_dataloader(fold, data_dir, used_column, args)
        
        # Debug: Check data structure
        for rally_batch, target in train_dataloader:
            print(f"rally_batch length: {len(rally_batch)}")
            for idx in range(len(rally_batch)):
                print(f"rally_batch[{idx}] shape: {rally_batch[idx].shape}")
            print(f"target shape: {target.shape}")
            break
        
        # Reset model parameters
        encoder.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
        encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

        train_auc_list, val_auc_list = [], []
        train_brier_list, val_brier_list = [], []
        train_loss_list, val_loss_list = [], []
        train_acc_list, val_acc_list = [], []
        best_fold_val_loss = float('inf')
        no_improve = 0
        max_length = train_dataloader.dataset.encode_length
        
        for epoch in tqdm(range(args['epochs'])):
            encoder.train()
            train_loss, train_preds, train_labels = 0.0, [], []
            for rally_batch, target in train_dataloader:
                rally_batch = [b.to(device) for b in rally_batch]
                target = target.to(device).float()
                encoder_optimizer.zero_grad()
                #player, shot_type,adj,player_A_loc,player_B_loc,mask
                win_logit,_ = encoder(
                    rally_batch[0].to(device), rally_batch[1].to(device), rally_batch[2].to(device),
                    rally_batch[3].to(device), rally_batch[4].to(device), rally_batch[5].to(device),
                    max_length
                )
                loss = bce_loss(win_logit, target)
                loss.backward()
                encoder_optimizer.step()
                train_loss += loss.item() * rally_batch[0].size(0)
                train_preds.extend(torch.sigmoid(win_logit).detach().cpu().numpy())
                train_labels.extend(target.detach().cpu().numpy())
                # Clear memory
                del win_logit, loss
                torch.cuda.empty_cache() if device == "cuda" else None
                gc.collect()
            train_loss /= len(train_dataloader.dataset)
            train_auc = roc_auc_score(train_labels, train_preds) if train_labels else float('nan')
            train_brier = brier_score_loss(train_labels, train_preds)
            train_acc = accuracy_score(train_labels, [1 if p >= 0.5 else 0 for p in train_preds])
            train_loss_list.append(train_loss)
            train_auc_list.append(train_auc)
            train_brier_list.append(train_brier)
            train_acc_list.append(train_acc)
            print(f'Fold {fold + 1} Epoch {epoch + 1} - Avg Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}, Train Brier: {train_brier:.4f}, Train Acc: {train_acc:.4f}')
            
            # Validation
            val_loss, val_auc, val_brier, val_acc = evaluate(valid_dataloader, encoder, args, device)
            val_loss_list.append(val_loss)
            val_auc_list.append(val_auc)
            val_brier_list.append(val_brier)
            val_acc_list.append(val_acc)
            print(f'Fold {fold + 1} Epoch {epoch + 1} - Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}, Val Brier: {val_brier:.4f}, Val Acc: {val_acc:.4f}')
            
            # Early stopping
            if val_loss < best_fold_val_loss:
                best_fold_val_loss = val_loss
                no_improve = 0
                output_folder_name = os.path.join(args['model_folder'], f'fold_{fold + 1}')
                if not os.path.exists(output_folder_name):
                    os.makedirs(output_folder_name)
                save_path = os.path.join(output_folder_name, 'encoder')
                torch.save(encoder.state_dict(), save_path)
                print(f"  ✔ Fold {fold + 1} New best model saved at: {save_path}")
                
                if best_fold_val_loss < best_val_loss:
                    best_val_loss = best_fold_val_loss
                    best_fold = fold + 1
            else:
                no_improve += 1
                print(f"  ⚠ Fold {fold + 1} No improvement for {no_improve}/{patience} epochs.")
                if no_improve >= patience:
                    print(f"🔚 Fold {fold + 1} Early stopping triggered.")
                    break
        
        # Save fold metrics
        fold_results['train_auc_list'].append(train_auc_list)
        fold_results['val_auc_list'].append(val_auc_list)
        fold_results['train_brier_list'].append(train_brier_list)
        fold_results['val_brier_list'].append(val_brier_list)
        fold_results['train_loss_list'].append(train_loss_list)
        fold_results['val_loss_list'].append(val_loss_list)
        fold_results['train_acc_list'].append(train_acc_list)
        fold_results['val_acc_list'].append(val_acc_list)
        
        # Plot metrics
        draw_plot(
            train_auc_list=train_auc_list,
            val_auc_list=val_auc_list,
            train_brier_list=train_brier_list,
            val_brier_list=val_brier_list,
            train_loss_list=train_loss_list,
            val_loss_list=val_loss_list,
            train_acc_list=train_acc_list,
            val_acc_list=val_acc_list,
            output_folder=os.path.join(args['model_folder'], f'fold_{fold + 1}')
        )
        
        # Save per-fold metrics to pickle
        with open(os.path.join(args['model_folder'], f'fold_{fold + 1}', 'fold_metrics.pkl'), 'wb') as f:
            pickle.dump({
                'train_auc_list': train_auc_list,
                'val_auc_list': val_auc_list,
                'train_brier_list': train_brier_list,
                'val_brier_list': val_brier_list,
                'train_loss_list': train_loss_list,
                'val_loss_list': val_loss_list,
                'train_acc_list': train_acc_list,
                'val_acc_list': val_acc_list
            }, f)
        
        # Record best validation metrics
        fold_val_losses.append(best_fold_val_loss)
        fold_val_aucs.append(max(val_auc_list))
        fold_val_briers.append(min(val_brier_list))
        fold_val_acc.append(max(val_acc_list))
        
        # Clear memory after fold
        del train_dataloader, valid_dataloader
        torch.cuda.empty_cache() if device == "cuda" else None
        gc.collect()
    
    # Compute average metrics
    avg_val_loss = np.mean(fold_val_losses)
    avg_val_auc = np.mean(fold_val_aucs)
    avg_val_brier = np.mean(fold_val_briers)
    avg_val_acc = np.mean(fold_val_acc)
    std_val_auc = np.std(fold_val_aucs)
    
    # Save all fold metrics
    with open(os.path.join(args['model_folder'], 'all_fold_metrics.pkl'), 'wb') as f:
        pickle.dump(fold_results, f)
    
    print(f"\nK-Fold Cross-Validation Results:")
    print(f"Average Val Loss: {avg_val_loss:.4f}")
    print(f"Average Val AUC: {avg_val_auc:.4f} (±{std_val_auc:.4f})")
    print(f"Average Val Brier: {avg_val_brier:.4f}")
    print(f"Average Val Acc: {avg_val_acc:.4f}")
    
    # Evaluate best model on test set
    print(f"\nEvaluating best model (Fold {best_fold}) on test dataset...")
    encoder.load_state_dict(torch.load(os.path.join(args['model_folder'], f'fold_{best_fold}', 'encoder')))
    test_loss, test_auc, test_brier, test_acc = evaluate(test_dataloader, encoder, args, device)

    return avg_val_loss, avg_val_auc, avg_val_brier, avg_val_acc, test_loss, test_auc, test_brier, test_acc

def evaluate(test_dataloader, encoder, args, device="cpu"):
    max_length = test_dataloader.dataset.encode_length
    bce_loss = BCEWithLogitsLoss()
    rally_analysis = []
    encoder.eval()
    with torch.no_grad():
        test_loss = 0.0
        n_test = 0
        y_true, y_prob = [], []
        for rally_batch, target in test_dataloader:
            target = target.to(device).float()
            #player, shot_type,adj,player_A_loc,player_B_loc,mask
            win_logit, node_attention_weights = encoder(
                rally_batch[0].to(device), rally_batch[1].to(device), rally_batch[2].to(device),
                rally_batch[3].to(device), rally_batch[4].to(device), rally_batch[5].to(device),
                max_length
            )
            y_prob.extend(torch.sigmoid(win_logit).detach().cpu().numpy())
            y_true.extend(target.cpu().numpy())
            l = bce_loss(win_logit, target)
            test_loss += l.item() * rally_batch[0].size(0)
            n_test += rally_batch[0].size(0)

            # 保存每個拉力的分析數據
            for i in range(rally_batch[0].size(0)):
                rally_data = {
                    'rally_id': i,
                    'node_attention': node_attention_weights[i].detach().cpu().numpy().tolist(),
                    # 'edge_attention': edge_attention_weights[i].detach().cpu().numpy().tolist(),
                    # 'edge_types': edge_types[i].detach().cpu().numpy().tolist(),
                    # 'edge_indices': edge_indices[i].detach().cpu().numpy().tolist(),
                    'playerID': rally_batch[0][i].detach().cpu().numpy().tolist(),
                    'shot_type': rally_batch[1][i].detach().cpu().numpy().tolist(),
                    'player_A_loc': rally_batch[3][i].detach().cpu().numpy().tolist(),
                    'player_B_loc': rally_batch[4][i].detach().cpu().numpy().tolist(),
                    'win_prob': torch.sigmoid(win_logit[i]).detach().cpu().numpy().item(),
                }
                rally_analysis.append(rally_data)

            del win_logit, l
            torch.cuda.empty_cache() if device == "cuda" else None
            gc.collect()
    avg_test_loss = test_loss / n_test if n_test else 0.0
    y_pred = [1 if p >= 0.5 else 0 for p in y_prob]
    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float('nan')
    brier = brier_score_loss(y_true, y_prob)

    # 保存分析結果
    output_folder_name = '/content/drive/MyDrive/DyMF_2025-07-22-02:32/'

    with open(os.path.join(output_folder_name, 'rally_analysis.json'), 'w') as f:
        json.dump(rally_analysis, f, indent=2)

    # # 可視化選手互動和擊球重要度
    # visualize_player_influence(rally_analysis, output_folder_name)
    visualize_shot_importance(rally_analysis, output_folder_name)
    
    return avg_test_loss, auc, brier, acc

def visualize_shot_importance(rally_analysis, output_folder):
    shot_types = {1: 'short service', 2: 'clear', 3: 'push & rush', 4: 'smash', 5: 'defensive return', 
                  6: 'drive', 7: 'net shot', 8: 'lob', 9: 'drop', 10: 'long service'}
    area_types = {
        1: "Front-Right", 2: "Front-Center", 3: "Back-Left", 4: "Middle-Left",
        5: "Back-Center", 6: "Middle-Center", 7: "Net Zone Center", 8: "Middle-Right",
        9: "Back-Right", 10: "Out"
    }
   
    # 只處理 rally_0
    rally = rally_analysis[0]  # 假設 rally_0 是第一個元素
    rally_id = rally['rally_id']
    node_attention = np.array(rally['node_attention'])
    shot_type = np.array(rally['shot_type'])
    playerID = np.array(rally['playerID'])

    # 過濾注意力權重 > 0 的擊球
    valid_indices = np.where((node_attention[0::2] > 0) | (node_attention[1::2] > 0))[0]
    if len(valid_indices) == 0:
        print(f"No shots with attention > 0 in rally {rally_id}, skipping visualization.")
        return

    shots = valid_indices+1
    attention_firstHitter = node_attention[0::2][valid_indices]  # 先攻者的注意力
    attention_secondHitter = node_attention[1::2][valid_indices]  # 後攻者的注意力
    shot_type_valid = shot_type[valid_indices]
    playerID_valid = playerID[valid_indices]

    # 判斷先攻者是 Player A (1) 還是 Player B (2)
    first_hitter = playerID_valid[0]  # 第一個 playerID 決定先攻者
    if first_hitter == 1:
        label_first = 'Player A'
        label_second = 'Player B'
        color_first = 'lightblue'
        color_second = 'lightgreen'
        attention_A = attention_firstHitter
        attention_B = attention_secondHitter
    else:  # first_hitter == 2
        label_first = 'Player B'
        label_second = 'Player A'
        color_first = 'lightgreen'
        color_second = 'lightblue'
        attention_A = attention_secondHitter
        attention_B = attention_firstHitter

    # 計算累積注意差異 (Player A - Player B)
    cumulative_diffA = np.cumsum(attention_A - attention_B)

    # 繪製擊球重要度柱狀圖
    plt.figure(figsize=(10, 6))
    plt.bar(shots - 0.2, attention_firstHitter, width=0.4, label=label_first, color=color_first)
    plt.bar(shots + 0.2, attention_secondHitter, width=0.4, label=label_second, color=color_second)
    plt.plot(shots,cumulative_diffA,  label='Cumulative Diff (A - B)', color='red', marker='o')
    plt.xlabel('Shot Number')
    plt.ylabel('Attention Weight')
    plt.title(f'Shot Importance Visualization')
    plt.grid(True)
    plt.legend()
    plt.xticks(shots)  # 確保 X 軸標籤從 1 開始
    plt.ylim(-0.1, 0.1)  # 固定 Y 軸範圍

    # 添加擊球類型標記，僅為實際擊球的玩家添加
    # for i, shot_idx in enumerate(shots):
    #     shot_name = shot_types.get(shot_type_valid[i], 'Unknown')
    #     if i % 2 == 0:  # First hitter's turn (0, 2, 4, ...)
    #         plt.text(shot_idx - 0.2, attention_firstHitter[i] + 0.01, f"{shot_name}", ha='center', fontsize=8)
    #     else:  # Second hitter's turn (1, 3, 5, ...)
    #         plt.text(shot_idx + 0.2, attention_secondHitter[i] + 0.01, f"{shot_name}", ha='center', fontsize=8)


    plt.savefig(os.path.join(output_folder, f'shot_importance_rally.png'))
    plt.close()

    

    # # 繪製 Win Influence 圖表
    # plt.figure(figsize=(10, 6))
    # plt.bar(shots, cumulative_diffA, width=0.4, label='Cumulative Diff (A - B)', color='blue', alpha=0.6)
    # plt.plot(shots,attention_firstHitter,  label=label_first, color=color_first, marker='o')
    # plt.plot(shots,attention_secondHitter, label=label_second, color=color_second, marker='o')
    # plt.xlabel('Shot Number')
    # plt.ylabel('Attention Score')
    # plt.title(f'Rally Win Influence (Player A Perspective)')
    # plt.grid(True)
    # plt.legend()
    # plt.xticks(shots)  # 確保 X 軸標籤從 1 開始
    # plt.ylim(-0.1, 0.1)  # 固定 Y 軸範圍

    # plt.savefig(os.path.join(output_folder, f'win_influence_rally.png'))
    # plt.close()

    print(f"Visualized rally with {len(shots)} shots.")


def visualize_final_shot_correlation(rally_analysis, output_folder):

    final_cum_diffs = []
    win_probs = []
    

    
    
    

    for rally in rally_analysis:
        rally_id = rally['rally_id']

        # 過濾注意力權重 > 0 的擊球
        valid_indices = np.where((node_attention[0::2] > 0) | (node_attention[1::2] > 0))[0]
        if len(valid_indices) == 0:
            print(f"No shots with attention > 0 in rally {rally_id}, skipping visualization.")
            return
    
        node_attention = np.array(rally['node_attention'])
        playerID = np.array(rally['playerID'])
        win_prob = rally['win_prob']

        shots = valid_indices+1
        attention_firstHitter = node_attention[0::2][valid_indices]  # 先攻者的注意力
        attention_secondHitter = node_attention[1::2][valid_indices]  # 後攻者的注意力

        # 判斷誰是 Player A
        first_hitter = playerID[0]
        if first_hitter == 1:
            attention_A = attention_firstHitter
            attention_B = attention_secondHitter
        else:
            attention_A = attention_secondHitter
            attention_B = attention_firstHitter

        # 累積差
        cumulative_diffA = np.cumsum(attention_A - attention_B)

        # 取最後一拍
        final_cum_diffs.append(cumulative_diffA[-1])
        win_probs.append(win_prob)

    final_cum_diffs = np.array(final_cum_diffs)
    win_probs = np.array(win_probs)

    # 計算 Pearson correlation
    r, p = pearsonr(final_cum_diffs, win_probs)

    # Scatter plot
    plt.figure(figsize=(7, 6))
    plt.scatter(final_cum_diffs, win_probs, alpha=0.6)
    plt.xlabel('Final Cumulative Attention Difference (A - B)')
    plt.ylabel('Predicted Win Probability')
    plt.title(f'Final Shot Attention vs Win Probability\nPearson r = {r:.3f}, p = {p:.3g}')
    plt.grid(True)

    plt.savefig(os.path.join(output_folder, 'final_shot_cumdiff_vs_winprob.png'))
    plt.close()

    print(f"Pearson r = {r:.4f}, p-value = {p:.4e}")

def save(encoder, decoder, args):
    output_folder_name = args['model_folder']
    if not os.path.exists(output_folder_name):
        os.makedirs(output_folder_name)
    torch.save(encoder.state_dict(), os.path.join(output_folder_name, 'encoder.pth'))
    torch.save(decoder.state_dict(), os.path.join(output_folder_name, 'decoder.pth'))