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
def train(train_dataloader, valid_dataloader, encoder, decoder, 
          location_criterion, shot_type_criterion, 
          encoder_optimizer, decoder_optimizer, args, device="cpu"):

    bce_loss = BCELoss()
    best_val_loss = float('inf')
    patience = args.get('patience', 5)
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
                max_length,
            )
            x_prob.extend(win_logit.detach().cpu().numpy())
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
                    valid_max_length,
                )
                y_prob.extend(win_logit.cpu().numpy())
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
 
    encoder.eval()
    y_true, y_prob = [], []
    total_loss = 0.0
    total_n = 0
    bce_loss = BCELoss()

    max_length = test_dataloader.dataset.encode_length

    with torch.no_grad():
        for rally_batch, target in test_dataloader:
            target = target.to(device).float()

            # forward
            win_logit = encoder(
                rally_batch[0].to(device),  # player
                rally_batch[1].to(device),  # shot_type
                rally_batch[2].to(device),  # player_A_x
                rally_batch[3].to(device),  # player_A_y
                rally_batch[4].to(device),  # player_B_x
                rally_batch[5].to(device),  # player_B_y
                rally_batch[7].to(device),  # （如果有其他輸入）
                max_length,                # encode 長度
            )

            # loss
            loss = bce_loss(win_logit, target)
            batch_size = target.size(0)
            total_loss += loss.item() * batch_size
            total_n += batch_size

            # 收集機率、label
            prob = torch.sigmoid(win_logit).cpu().numpy()
            y_prob.extend(prob.tolist())
            y_true.extend(target.cpu().numpy().tolist())

    # 平均 loss
    avg_loss = total_loss / total_n if total_n else float("nan")

    # classification metrics
    y_pred = [1 if p >= 0.5 else 0 for p in y_prob]
    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    brier = brier_score_loss(y_true, y_prob)

    # 列印
    print(f"Test Loss: {avg_loss:.4f} | Acc: {acc:.4f} | AUC: {auc:.4f} | Brier: {brier:.4f}")

    return avg_loss, acc, auc, brier

def predict(all_dataloader, encoder, decoder, args, device="cpu"):
    encode_length = args['encode_length']
    encoder.eval(), decoder.eval()

    all_player_A_x_record = []
    all_player_A_y_record = []
    all_player_B_x_record = []
    all_player_B_y_record = []
    all_shot_type_record = []

    all_mean_A = []
    all_mean_B = []
    all_cov_A = []
    all_cov_B = []
    
    mean_x, std_x = 175., 82.
    mean_y, std_y = 467., 192.

    with torch.no_grad():
        for rally, _ in tqdm(all_dataloader):
            
            player_A_x_record = []
            player_A_y_record = []
            player_B_x_record = []
            player_B_y_record = []
            shot_type_record = []

            # rally information
            player = rally[0].to(device).long()
            shot_type = rally[1].to(device).long()
            player_A_x = rally[2].to(device)
            player_A_y = rally[3].to(device)
            player_B_x = rally[4].to(device)
            player_B_y  = rally[5].to(device)

            encoder_player = player[:, 0:2]
            encoder_shot_type = shot_type[:, :encode_length-1]
            encoder_player_A_x = player_A_x[:, :encode_length]
            encoder_player_A_y = player_A_y[:, :encode_length]
            encoder_player_B_x = player_B_x[:, :encode_length]
            encoder_player_B_y = player_B_y[:, :encode_length]

            for i in range(encoder_player_A_x.size(1)):
                player_A_x_record.append(encoder_player_A_x[0][i].item() * std_x + mean_x)
                player_A_y_record.append(960 - (encoder_player_A_y[0][i].item() * std_y + mean_y))
                player_B_x_record.append(encoder_player_B_x[0][i].item() * std_x + mean_x)
                player_B_y_record.append(960 - (encoder_player_B_y[0][i].item() * std_y + mean_y))
            for i in range(encoder_shot_type.size(1)):
                shot_type_record.append(encoder_shot_type[0][i].item())


            encode_node_embedding, original_embedding, adjacency_matrix = encoder(encoder_player, encoder_shot_type, encoder_player_A_x, encoder_player_A_y, encoder_player_B_x, encoder_player_B_y, encode_length)
                
            decoder_node_embedding = encode_node_embedding.clone()
                        
            decoder_player = player[:, 0:2]

            decoder_player_A_x = player_A_x[:, encode_length-1:encode_length]
            decoder_player_A_y = player_A_y[:, encode_length-1:encode_length]
            decoder_player_B_x = player_B_x[:, encode_length-1:encode_length]
            decoder_player_B_y = player_B_y[:, encode_length-1:encode_length]
            
            first = True
            for sequence_index in range(encode_length, encode_length+1):
                predict_xy, predict_shot_type_logit, adjacency_matrix, decoder_node_embedding, original_embedding = decoder(decoder_player, sequence_index+1, decoder_node_embedding, original_embedding, adjacency_matrix, 
                                                                                                                        decoder_player_A_x, decoder_player_A_y, decoder_player_B_x, decoder_player_B_y,
                                                                                                                        shot_type=None, train=False, first=first)

                predict_A_xy = predict_xy[:, 0:1, :]
                predict_B_xy = predict_xy[:, 1:2, :]
                
                sx = torch.exp(predict_A_xy[:, -1, 2]) #sx
                sy = torch.exp(predict_A_xy[:, -1, 3]) #sy
                corr = torch.tanh(predict_A_xy[:, -1, 4]) #corr                

                cov = torch.zeros(2, 2).to(player.device)
                cov[0, 0]= sx * sx
                cov[0, 1]= corr * sx * sy
                cov[1, 0]= corr * sx * sy
                cov[1, 1]= sy * sy
                mean = predict_A_xy[:, -1, 0:2]
                
                mean_A = mean.cpu().detach().numpy()
                mean_A[0][0] = mean_A[0][0] * std_x + mean_x 
                mean_A[0][1] = 960 - (mean_A[0][1] * std_y + mean_y )
                cov_A = cov.cpu().detach().numpy()
                cov_A = cov_A * std_x * std_y

                all_mean_A.append(mean_A)            
                all_cov_A.append(cov_A)

                sx = torch.exp(predict_B_xy[:, -1, 2]) #sx
                sy = torch.exp(predict_B_xy[:, -1, 3]) #sy
                corr = torch.tanh(predict_B_xy[:, -1, 4]) #corr                

                cov = torch.zeros(2, 2).to(player.device)
                cov[0, 0]= sx * sx
                cov[0, 1]= corr * sx * sy
                cov[1, 0]= corr * sx * sy
                cov[1, 1]= sy * sy
                mean = predict_B_xy[:, -1, 0:2]
                
                mean_B = mean.cpu().detach().numpy()
                mean_B[0][0] = mean_B[0][0] * std_x + mean_x 
                mean_B[0][1] = 960 - (mean_B[0][1] * std_y + mean_y )
                cov_B = cov.cpu().detach().numpy()
                cov_B = cov_B * std_x * std_y

                all_mean_B.append(mean_B)
                all_cov_B.append(cov_B)             

                decoder_player_A_x = predict_A_xy[:, 0, 0:1]
                decoder_player_A_y = predict_A_xy[:, 0, 1:2]
                decoder_player_B_x = predict_B_xy[:, 0, 0:1]
                decoder_player_B_y = predict_B_xy[:, 0, 1:2]

                weights = predict_shot_type_logit[0, 1:]
                weights = F.softmax(weights, dim=0).cpu().detach().numpy()
                shot_type_record.append(weights)

            all_player_A_x_record.append(player_A_x_record)
            all_player_A_y_record.append(player_A_y_record)
            all_player_B_x_record.append(player_B_x_record)
            all_player_B_y_record.append(player_B_y_record)
            all_shot_type_record.append(shot_type_record)

    return all_player_A_x_record, all_player_A_y_record, all_player_B_x_record, all_player_B_y_record, all_shot_type_record, all_mean_A, all_mean_B, all_cov_A, all_cov_B

def save(encoder, decoder, args):
    output_folder_name = args['model_folder']
    if not os.path.exists(output_folder_name):
        os.makedirs(output_folder_name)
    
    torch.save(encoder.state_dict(), output_folder_name + '/encoder')
    torch.save(decoder.state_dict(), output_folder_name + '/decoder')
