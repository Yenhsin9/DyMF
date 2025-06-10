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
from sklearn.linear_model import LogisticRegression
import numpy as np
PAD = 0

def Gaussian2D_loss(V_pred, V_trgt):
    #mux, muy, sx, sy, corr
    #assert V_pred.shape == V_trgt.shape
    normx = V_trgt[:, 0] - V_pred[:, 0]
    normy = V_trgt[:, 1] - V_pred[:, 1]

    sx = torch.exp(V_pred[:, 2]) #sx
    sy = torch.exp(V_pred[:, 3]) #sy
    corr = torch.tanh(V_pred[:, 4]) #corr
    
    sxsy = sx * sy

    z = (normx/sx)**2 + (normy/sy)**2 - 2*((corr*normx*normy)/sxsy)
    negRho = 1 - corr**2

    # Numerator
    result = torch.exp(-z/(2*negRho))
    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

    # Final PDF calculation
    result = result / denom

    # Numerical stability
    epsilon = 1e-20

    result = -torch.log(torch.clamp(result, min=epsilon))
    result = torch.sum(result)
    
    return result

def train_kfold(fold_datasets,test_dataloader, encoder, location_criterion, shot_type_criterion, encoder_optimizer, args, device="cpu"):
    bce_loss = BCEWithLogitsLoss()
    best_val_loss = float('inf')
    patience = args.get('patience', 5)
    
    # 儲存每個折的指標
    fold_val_losses = []
    fold_val_aucs = []
    fold_val_briers = []
    best_fold = 0  # 記錄最佳折
    
    for fold, (train_dataloader, valid_dataloader, args) in enumerate(fold_datasets):
        print(f"\nTraining Fold {fold + 1}/{len(fold_datasets)}")
        
        # 重置模型參數
        encoder.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
        encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=args['lr'], weight_decay=0.00006)
        
        train_auc_list = []
        val_auc_list = []
        train_brier_list = []
        val_brier_list = []
        best_fold_val_loss = float('inf')
        no_improve = 0
        
        max_length = train_dataloader.dataset.encode_length
        
        for epoch in tqdm(range(args['epochs'])):
            train_loss = 0.0
            n_train = 0
            x_true, x_prob = [], []
            encoder.train()
            
            for rally_batch, target in train_dataloader:
                encoder_optimizer.zero_grad()
                target = target.to(device).float()
                win_logit = encoder(
                    rally_batch[0].to(device),
                    rally_batch[1].to(device),
                    rally_batch[2].to(device),
                    rally_batch[3].to(device),
                    rally_batch[4].to(device),
                    rally_batch[5].to(device),
                    rally_batch[7].to(device),
                    rally_batch[8].to(device),
                    rally_batch[9].to(device),
                    rally_batch[10].to(device),
                    rally_batch[11].to(device),
                    rally_batch[12].to(device),
                    max_length,
                )
                
                x_prob.extend(torch.sigmoid(win_logit).detach().cpu().numpy())
                x_true.extend(target.detach().cpu().numpy())
                loss = bce_loss(win_logit, target)
                loss.backward()
                encoder_optimizer.step()

                train_loss += loss.item() * rally_batch[0].size(0)
                n_train += rally_batch[0].size(0)

            avg_train_loss = train_loss / n_train if n_train else 0.0
            x_pred = [1 if p >= 0.5 else 0 for p in x_prob]
            try:
                T_auc = roc_auc_score(x_true, x_prob)
            except ValueError:
                T_auc = float('nan')
            T_brier = brier_score_loss(x_true, x_prob)
            train_auc_list.append(T_auc)
            train_brier_list.append(T_brier)
            print(f'Fold {fold + 1} Epoch {epoch + 1} - Avg Train Loss: {avg_train_loss:.4f}, Train AUC: {T_auc:.4f}, Train Brier: {T_brier:.4f}')

            # 驗證階段
            encoder.eval()
            with torch.no_grad():
                val_loss = 0.0
                n_val = 0
                y_true, y_prob = [], []
                valid_max_length = valid_dataloader.dataset.encode_length
                for rally_batch, target in valid_dataloader:
                    target = target.to(device).float()
                    win_logit = encoder(
                        rally_batch[0].to(device),
                        rally_batch[1].to(device),
                        rally_batch[2].to(device),
                        rally_batch[3].to(device),
                        rally_batch[4].to(device),
                        rally_batch[5].to(device),
                        rally_batch[7].to(device),
                        rally_batch[8].to(device),
                        rally_batch[9].to(device),
                        rally_batch[10].to(device),
                        rally_batch[11].to(device),
                        rally_batch[12].to(device),
                        valid_max_length,
                    )
                    y_prob.extend(torch.sigmoid(win_logit).detach().cpu().numpy())
                    y_true.extend(target.cpu().numpy())

                    l = bce_loss(win_logit, target)
                    val_loss += l.item() * rally_batch[0].size(0)
                    n_val += rally_batch[0].size(0)

            avg_val_loss = val_loss / n_val if n_val else 0.0
            y_pred = [1 if p >= 0.5 else 0 for p in y_prob]
            acc = accuracy_score(y_true, y_pred)
            try:
                auc = roc_auc_score(y_true, y_prob)
            except ValueError:
                auc = float('nan')
            brier = brier_score_loss(y_true, y_prob)
            val_auc_list.append(auc)
            val_brier_list.append(brier)
            print(
                f"Fold {fold + 1} Epoch {epoch + 1} - "
                f"Val Loss: {avg_val_loss:.4f}, "
                f"Acc: {acc:.4f}, "
                f"Val AUC: {auc:.4f}, "
                f"Val Brier: {brier:.4f}"
            )
            
            # 早停
            if avg_val_loss < best_fold_val_loss:
                best_fold_val_loss = avg_val_loss
                no_improve = 0
                output_folder_name = os.path.join(args['model_folder'], f'fold_{fold + 1}')
                if not os.path.exists(output_folder_name):
                    os.makedirs(output_folder_name)
                torch.save(encoder.state_dict(), f"{output_folder_name}/encoder")
                print(f"  ✔ Fold {fold + 1} New best model saved.")
                
                # 更新全局最佳折
                if best_fold_val_loss < best_val_loss:
                    best_val_loss = best_fold_val_loss
                    best_fold = fold + 1
            else:
                no_improve += 1
                print(f"  ⚠ Fold {fold + 1} No improvement for {no_improve}/{patience} epochs.")
                if no_improve >= patience:
                    print(f"🔚 Fold {fold + 1} Early stopping triggered.")
                    break
        
        # 記錄該折的最佳驗證指標
        fold_val_losses.append(best_fold_val_loss)
        fold_val_aucs.append(max(val_auc_list))
        fold_val_briers.append(min(val_brier_list))
        
    # 計算所有折的平均指標
    avg_val_loss = np.mean(fold_val_losses)
    avg_val_auc = np.mean(fold_val_aucs)
    avg_val_brier = np.mean(fold_val_briers)
    std_val_auc = np.std(fold_val_aucs)
    
    print(f"\nK-Fold Cross-Validation Results:")
    print(f"Average Val Loss: {avg_val_loss:.4f}")
    print(f"Average Val AUC: {avg_val_auc:.4f} (±{std_val_auc:.4f})")
    print(f"Average Val Brier: {avg_val_brier:.4f}")
    
    # 在測試集上評估最佳折的模型
    print(f"\nEvaluating best model (Fold {best_fold}) on test dataset...")
    encoder.load_state_dict(torch.load(os.path.join(args['model_folder'], f'fold_{best_fold}/encoder')))
    test_loss, test_auc, test_brier = evaluate(test_dataloader, encoder, args, device)
    
    return avg_val_loss, avg_val_auc, avg_val_brier,test_loss, test_auc, test_brier

def train(train_dataloader, valid_dataloader, encoder,
          location_criterion, shot_type_criterion, 
          encoder_optimizer, args, device="cpu"):

    bce_loss = BCEWithLogitsLoss()
    best_val_loss = float('inf')
    patience = args.get('patience', 10)
    no_improve = 0
    max_length = train_dataloader.dataset.encode_length
    train_auc_list=[]
    val_auc_list=[]  
    train_brier_list=[]  
    val_brier_list=[]  
  
    for epoch in tqdm(range(args['epochs'])):
        train_loss = 0.0
        n_train    = 0
        x_true, x_prob = [], []
        encoder.train() 
        for rally_batch, target in train_dataloader:
            encoder_optimizer.zero_grad()
            target = target.to(device).float()
            win_logit  = encoder(
                rally_batch[0].to(device),
                rally_batch[1].to(device),
                rally_batch[2].to(device),
                rally_batch[3].to(device),
                rally_batch[4].to(device),
                rally_batch[5].to(device),
                rally_batch[7].to(device),
                rally_batch[8].to(device),
                rally_batch[9].to(device),
                rally_batch[10].to(device),
                rally_batch[11].to(device),
                rally_batch[12].to(device),
                max_length,
            )
            
            x_prob.extend(torch.sigmoid(win_logit).detach().cpu().numpy())
            x_true.extend(target.detach().cpu().numpy())
            loss = bce_loss(win_logit, target)
            loss.backward()
            encoder_optimizer.step()

            train_loss += loss.item() * rally_batch[0].size(0)
            n_train += rally_batch[0].size(0)

        avg_train_loss = train_loss / n_train if n_train else 0.0
        x_pred = [1 if p >= 0.5 else 0 for p in x_prob]
        T_acc = accuracy_score(x_true, x_pred)
        try:
            T_acc = roc_auc_score(x_true, x_prob)
        except ValueError:
            T_acc = float('nan')
        # 計算 Brier score
        T_brier = brier_score_loss(x_true, x_prob)
        train_auc_list.append(T_acc)
        train_brier_list.append(T_brier)
        print('avg_train_loss',avg_train_loss)

        valid_max_length = valid_dataloader.dataset.encode_length
        # Validation phase
        encoder.eval()
        with torch.no_grad():
            val_loss = 0.0
            n_val = 0
            y_true, y_prob = [], []
            for rally_batch, target in valid_dataloader:
                target = target.to(device).float()
                win_logit = encoder(
                    rally_batch[0].to(device),
                    rally_batch[1].to(device),
                    rally_batch[2].to(device),
                    rally_batch[3].to(device),
                    rally_batch[4].to(device),
                    rally_batch[5].to(device),
                    rally_batch[7].to(device),
                    rally_batch[8].to(device),
                    rally_batch[9].to(device),
                    rally_batch[10].to(device),
                    rally_batch[11].to(device),
                    rally_batch[12].to(device),
                    valid_max_length,
                )
                y_prob.extend(torch.sigmoid(win_logit).detach().cpu().numpy())
                y_true.extend(target.cpu().numpy())

                l = bce_loss(win_logit, target)
                val_loss += l.item() * rally_batch[0].size(0)
                n_val += rally_batch[0].size(0)

        avg_val_loss = val_loss / n_val if n_val else 0.0
        y_pred = [1 if p >= 0.5 else 0 for p in y_prob]
        acc = accuracy_score(y_true, y_pred)
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = float('nan')
        # 計算 Brier score
        brier = brier_score_loss(y_true, y_prob)
        val_auc_list.append(auc)
        val_brier_list.append(brier)
        print(
            f"Epoch {epoch+1}/{args['epochs']} - "
            f"processed {n_train} rallies: "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"Acc: {acc:.4f}, "
            f"Train AUC: {T_acc:.4f}, "
            f"Val AUC: {auc:.4f}, "
            f"Train Brier: {T_brier:.4f}, "
            f"Val Brier: {brier:.4f}"
        )
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve = 0
            # Save best model
            output_folder_name = args['model_folder']
            if not os.path.exists(output_folder_name):
                os.makedirs(output_folder_name)
            torch.save(encoder.state_dict(), output_folder_name + '/encoder')
            print("  ✔ New best model saved.")
        else:
            no_improve += 1
            print(f"  ⚠ No improvement for {no_improve}/{patience} epochs.")
            if no_improve >= patience:
                print("🔚 Early stopping triggered.")
                break
    draw_plot(train_auc_list,val_auc_list,train_brier_list,val_brier_list)
    return best_val_loss

def evaluate(test_dataloader,
             encoder,
             args,
             device="cpu"):
    
    max_length = test_dataloader.dataset.encode_length
    bce_loss = BCEWithLogitsLoss()
    test_auc_list=[]  
    test_brier_list=[] 

    encoder.eval()
    with torch.no_grad():
        test_loss = 0.0
        n_test = 0
        y_true, y_prob = [], []
        for rally_batch, target in test_dataloader:
            target = target.to(device).float()

            # forward
            win_logit = encoder(
                rally_batch[0].to(device),
                rally_batch[1].to(device),
                rally_batch[2].to(device),
                rally_batch[3].to(device),
                rally_batch[4].to(device),
                rally_batch[5].to(device),
                rally_batch[7].to(device),
                rally_batch[8].to(device),
                rally_batch[9].to(device),
                rally_batch[10].to(device),
                rally_batch[11].to(device),
                rally_batch[12].to(device),
                max_length,
            )
            y_prob.extend(torch.sigmoid(win_logit).detach().cpu().numpy())
            y_true.extend(target.cpu().numpy())

            l = bce_loss(win_logit, target)
            test_loss += l.item() * rally_batch[0].size(0)
            n_test += rally_batch[0].size(0)

    avg_test_loss = test_loss / n_test if n_test else 0.0
    y_pred = [1 if p >= 0.5 else 0 for p in y_prob]
    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float('nan')
    # 計算 Brier score
    brier = brier_score_loss(y_true, y_prob)
    test_auc_list.append(auc)
    test_brier_list.append(brier)
    print(
        f"Test Loss: {avg_test_loss:.4f}, "
        f"Acc: {acc:.4f}, "
        f"Test AUC: {auc:.4f}, "
        f"Test Brier: {brier:.4f}"
    )

    return avg_test_loss, auc, brier

def save(encoder, decoder, args):
    output_folder_name = args['model_folder']
    if not os.path.exists(output_folder_name):
        os.makedirs(output_folder_name)
    
    torch.save(encoder.state_dict(), output_folder_name + '/encoder')
    torch.save(decoder.state_dict(), output_folder_name + '/decoder')
