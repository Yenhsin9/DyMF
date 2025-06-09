import argparse
import torch
import numpy as np
import random
import torch.nn as nn
from datetime import datetime
import os
import optuna

from prepare_dataset import prepare_dataset
from utils import save_args_file

def objective(trial, args):
    # Suggest hyperparameters to tune
    args['lr'] = trial.suggest_float('lr', 0.001, 0.006, log=True)
    args['hidden_size'] = trial.suggest_int('hidden_size', 8, 16, step=16)
    args['dropout'] = trial.suggest_float('dropout', 0.1, 0.5)
    args['weight_decay'] = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)

    # Set random seeds for reproducibility
    np.random.seed(args['seed'])
    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed(args['seed'])
    torch.cuda.manual_seed_all(args['seed'])
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Prepare dataset
    train_dataloader, valid_dataloader, test_dataloader, args = prepare_dataset(args)
    TrainMAXlength = train_dataloader.dataset.encode_length
    ValMAXlength = valid_dataloader.dataset.encode_length
    args['max_length'] = max(ValMAXlength, TrainMAXlength)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"使用設備：{device}")
    # Initialize model based on model_type
    if args['model_type'] == 'DNRI':
        from DNRI.model import Encoder
        from DNRI.runner import train
        encoder = Encoder(args)

    elif args['model_type'] == 'LSTM':
        from LSTM.model import Encoder, Decoder
        from LSTM.runner import train
        encoder = Encoder(args)
        decoder = Decoder(args)
        encoder.player_embedding.weight = decoder.player_embedding.weight
        encoder.type_embedding.weight = decoder.type_embedding.weight
        encoder.coordination_transform.weight = decoder.coordination_transform.weight

    elif args['model_type'] == 'DyMF':
        if args['use_complete_graph'] == 1:
            from DyMF.model_complete import Encoder
            from DyMF.runner import train
        elif args['without_dynamic_gcn'] == 1:
            from DyMF.model_without_dynamic_gcn import Encoder
            from DyMF.runner import train
        elif args['without_tactical_fusion'] == 1:
            from DyMF.model_without_tactical_fusion import Encoder
            from DyMF.runner import train
        elif args['without_player_style_fusion'] == 1:
            from DyMF.model_without_player_style_fusion import Encoder
            from DyMF.runner import train
        elif args['without_rally_fusion'] == 1:
            from DyMF.model_without_rally_fusion import Encoder
            from DyMF.runner import train
        elif args['without_style_fusion'] == 1:
            from DyMF.model_without_style_fusion import Encoder
            from DyMF.runner import train
        else:
            from DyMF.model import Encoder
            from DyMF.runner import train
        encoder = Encoder(args, device)

    elif args['model_type'] == 'GCN':
        if args['use_complete_graph'] == 1:
            from GCN.model_complete import Encoder, Decoder
        else:
            from GCN.model import Encoder, Decoder
        from GCN.runner import train
        encoder = Encoder(args)
        decoder = Decoder(args)
        encoder.player_embedding.weight = decoder.player_embedding.weight
        encoder.coordination_transform.weight = decoder.coordination_transform.weight

    elif args['model_type'] == 'ShuttleNet':
        from ShuttleNet.ShuttleNet import ShotGenEncoder, ShotGenPredictor
        from ShuttleNet.runner import train
        encoder = ShotGenEncoder(args)
        decoder = ShotGenPredictor(args)
        encoder.player_embedding.weight = decoder.shotgen_decoder.player_embedding.weight
        encoder.type_embedding.weight = decoder.shotgen_decoder.type_embedding.weight
        encoder.coordination_transform.weight = decoder.shotgen_decoder.coordination_transform.weight

    elif args['model_type'] == 'rGCN':
        if args['use_complete_graph'] == 1:
            from rGCN.model_complete import Encoder, Decoder
        else:
            from rGCN.model import Encoder, Decoder
        from rGCN.runner import train
        encoder = Encoder(args, device)
        decoder = Decoder(args, device)
        encoder.player_embedding.weight = decoder.player_embedding.weight
        encoder.coordination_transform.weight = decoder.coordination_transform.weight

    elif args['model_type'] == 'Transformer':
        from Transformer.transformer import TransformerEncoder, TransformerPredictor
        from Transformer.runner import train
        encoder = TransformerEncoder(args)
        decoder = TransformerPredictor(args)
        encoder.player_embedding.weight = decoder.transformer_decoder.player_embedding.weight
        encoder.type_embedding.weight = decoder.transformer_decoder.type_embedding.weight
        encoder.coordination_transform.weight = decoder.transformer_decoder.coordination_transform.weight

    elif args['model_type'] == 'GCN_d':
        if args['use_complete_graph'] == 1:
            from GCN_dynamic.model_complete import Encoder, Decoder
        else:
            from GCN_dynamic.model import Encoder, Decoder
        from GCN_dynamic.runner import train
        encoder = Encoder(args, device)
        decoder = Decoder(args, device)
        encoder.player_embedding.weight = decoder.player_embedding.weight
        encoder.coordination_transform.weight = decoder.coordination_transform.weight

    elif args['model_type'] == 'eGCN':
        if args['use_complete_graph'] == 1:
            from eGCN.model_complete import Encoder, Decoder
        else:
            from eGCN.model import Encoder, Decoder
        from eGCN.runner import train
        encoder = Encoder(args)
        decoder = Decoder(args)
        encoder.player_embedding.weight = decoder.player_embedding.weight
        encoder.coordination_transform.weight = decoder.coordination_transform.weight

    # Optimizer with weight decay
    encoder_optimizer = torch.optim.Adam(
        encoder.parameters(),
        lr=args['lr'],
        weight_decay=args['weight_decay']
    )

    location_criterion = nn.MSELoss()
    shot_type_criterion = nn.CrossEntropyLoss()

    encoder.to(device), location_criterion.to(device), shot_type_criterion.to(device)

    total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    # Train and get best validation loss
    best_val_loss = train(train_dataloader, valid_dataloader, encoder, location_criterion, shot_type_criterion, encoder_optimizer, args, device=device)

    # Save args for this trial
    output_folder_name = os.path.join(args['model_folder'], f"trial_{trial.number}")
    if not os.path.exists(output_folder_name):
        os.makedirs(output_folder_name)
    torch.save(encoder.state_dict(), output_folder_name + '/encoder')

    print(f"Trial {trial.number} - Best Val Loss: {best_val_loss:.4f}")
    return best_val_loss

def main():
    args = argparse.ArgumentParser()

    # Prepare data
    args.add_argument("--input_data_folder_path", type=str, default="./data/")
    args.add_argument("--match_list_csv", type=str, default="match.csv")
    args.add_argument("--homography_matrix_list_csv", type=str, default="homography.csv")
    args.add_argument("--prepared_data_output_path", type=str, default="./data/dataset.csv")
    args.add_argument("--already_have_data", type=int, default=1)
    args.add_argument("--preprocessed_data_path", type=str, default="./data/dataset.csv")
    args.add_argument("--train_ratio", type=float, default=0.7)
    args.add_argument("--valid_ratio", type=float, default=0.15)
    args.add_argument("--max_length", type=int, default=100)

    # Training (some hyperparameters will be set by Optuna)
    args.add_argument("--seed", type=int, default=22)
    args.add_argument("--train_batch_size", type=int, default=32)
    args.add_argument("--valid_batch_size", type=int, default=8)
    args.add_argument("--test_batch_size", type=int, default=8)
    args.add_argument("--player_dim", type=int, default=16)
    args.add_argument("--type_dim", type=int, default=16)
    args.add_argument("--location_dim", type=int, default=16)
    args.add_argument("--num_layer", type=int, default=2)
    args.add_argument("--epochs", type=int, default=30)
    args.add_argument("--num_basis", type=int, default=2)
    args.add_argument("--model_type", type=str, required=True)

    # Ablation
    args.add_argument("--use_complete_graph", type=int, default=0)
    args.add_argument("--without_dynamic_gcn", type=int, default=0)
    args.add_argument("--without_tactical_fusion", type=int, default=0)
    args.add_argument("--without_player_style_fusion", type=int, default=0)
    args.add_argument("--without_rally_fusion", type=int, default=0)
    args.add_argument("--without_style_fusion", type=int, default=0)
    args.add_argument("--without_refer", type=int, default=0)

    # Save model
    args.add_argument("--output_model_path", type=str, default='./model/')
    args.add_argument("--model_folder", type=str, default=None)

    # Sample
    args.add_argument("--sample_num", type=int, default=1)

    # Optuna trials
    args.add_argument("--n_trials", type=int, default=10)

    args = args.parse_args()
    args = vars(args)
    if args['model_folder'] == None:
        args['model_folder'] = './model/' +  args['model_type'] + '_' + str(datetime.now().strftime("%Y-%m-%d-%H:%M"))

    # Create Optuna study
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, args), n_trials=args['n_trials'])

    # Print best trial
    best_trial = study.best_trial
    print(f"Best trial: {best_trial.number}")
    print(f"Best validation loss: {best_trial.value:.4f}")
    print("Best hyperparameters: ")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")

    # Update args with best hyperparameters
    args.update(best_trial.params)

    # Train final model with best hyperparameters
    train_dataloader, valid_dataloader, test_dataloader, args = prepare_dataset(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Re-initialize model with best hyperparameters
    if args['model_type'] == 'DyMF':
        from DyMF.model import Encoder
        from DyMF.runner import train
        encoder = Encoder(args, device)
    else:
        raise ValueError(f"Model type {args['model_type']} not supported in this example.")

    encoder_optimizer = torch.optim.Adam(
        encoder.parameters(),
        lr=args['lr'],
        weight_decay=args['weight_decay']
    )

    location_criterion = nn.MSELoss()
    shot_type_criterion = nn.CrossEntropyLoss()

    encoder.to(device), location_criterion.to(device), shot_type_criterion.to(device)

    # Train final model
    best_val_loss = train(train_dataloader, valid_dataloader, encoder, location_criterion, shot_type_criterion, encoder_optimizer, args, device=device)

    # Save final model
    output_folder_name = os.path.join(args['model_folder'], f"final_model")
    if not os.path.exists(output_folder_name):
        os.makedirs(output_folder_name)
    torch.save(encoder.state_dict(), output_folder_name + '/encoder')
    print(f"Final best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()