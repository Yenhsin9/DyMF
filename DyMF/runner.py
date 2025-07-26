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
from prepare_dataset import get_fold_dataloader
import gc
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
                win_logit = encoder(
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
    encoder.eval()
    with torch.no_grad():
        test_loss = 0.0
        n_test = 0
        y_true, y_prob = [], []
        for rally_batch, target in test_dataloader:
            target = target.to(device).float()
            #player, shot_type,adj,player_A_loc,player_B_loc,mask
            win_logit = encoder(
                rally_batch[0].to(device), rally_batch[1].to(device), rally_batch[2].to(device),
                rally_batch[3].to(device), rally_batch[4].to(device), rally_batch[5].to(device),
                max_length
            )
            y_prob.extend(torch.sigmoid(win_logit).detach().cpu().numpy())
            y_true.extend(target.cpu().numpy())
            l = bce_loss(win_logit, target)
            test_loss += l.item() * rally_batch[0].size(0)
            n_test += rally_batch[0].size(0)
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
    
    return avg_test_loss, auc, brier, acc

def save(encoder, decoder, args):
    output_folder_name = args['model_folder']
    if not os.path.exists(output_folder_name):
        os.makedirs(output_folder_name)
    torch.save(encoder.state_dict(), os.path.join(output_folder_name, 'encoder.pth'))
    torch.save(decoder.state_dict(), os.path.join(output_folder_name, 'decoder.pth'))