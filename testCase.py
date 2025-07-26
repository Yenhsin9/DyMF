import argparse
import sys
import torch
import torch.nn as nn
import random
import numpy as np
import pandas as pd
from prepare_dataset import prepare_kfold_datasets
from utils import load_args_file

import os


def main():
    args = argparse.ArgumentParser()
    # prepare data
    args.add_argument("--input_data_folder_path", type=str, default="./data/")
    args.add_argument("--match_list_csv", type=str, default="match.csv")
    args.add_argument("--homography_matrix_list_csv", type=str, default="homography.csv")
    args.add_argument("--prepared_data_output_path", type=str, default="./data/dataset.csv")
    args.add_argument("--already_have_data", type=int, default=1) #if have data
    args.add_argument("--preprocessed_data_path", type=str, default="./data/dataset.csv")
    args.add_argument("--train_ratio", type=float, default=0.6)
    args.add_argument("--valid_ratio", type=float, default=0.2)
    args.add_argument("--test_ratio", type=float, default=0.2)
    args.add_argument("--max_length", type=int, default=100)

    # training
    args.add_argument("--seed", type=int, default=22)
    args.add_argument("--train_batch_size", type=int, default=64)
    args.add_argument("--valid_batch_size", type=int, default=32) 
    args.add_argument("--test_batch_size", type=int, default=32)
    args.add_argument("--hidden_size", type=int, default=64)
    args.add_argument("--model_type", type=str, default='DyMF')
    args.add_argument("--lr", type=float, default=0.0019)
    args.add_argument("--player_dim", type=int, default=16)
    args.add_argument("--type_dim", type=int, default=16)
    args.add_argument("--location_dim", type=int, default=16)
    args.add_argument("--num_layer", type=int, default=2)
    args.add_argument("--weight_decay", type=float, default=0.000147)

    args.add_argument("--epochs", type=int, default=50)
    #args.add_argument("--encode_length", type=int, required=True)
    args.add_argument("--dropout", type=float, default=0.69723)

    args.add_argument("--num_basis", type=int, default=3)

    # ablation
    args.add_argument("--use_complete_graph", type=int, default=0)
    args.add_argument("--without_dynamic_gcn", type=int, default=0)
    args.add_argument("--without_tactical_fusion", type=int, default=0)
    args.add_argument("--without_player_style_fusion", type=int, default=0)
    args.add_argument("--without_rally_fusion", type=int, default=0)
    args.add_argument("--without_style_fusion", type=int, default=0)
    args.add_argument("--without_refer", type=int, default=0)

    # save model
    args.add_argument("--output_model_path", type=str, default='./model/')
    args.add_argument("--model_folder", type=str, default=None)

    # sample
    args.add_argument("--sample_num", type=int, default=1)

    # k-fold 參數
    args.add_argument("--k_folds", type=int, default=5)

    args = args.parse_args()
    args = vars(args)

    # Set random seeds for reproducibility
    np.random.seed(args['seed'])
    random.seed(args['seed'])
    torch.manual_seed(args['seed'])

    # Device-specific seed and settings
    if torch.backends.mps.is_available():
        # MPS-specific seed (available in PyTorch 2.3+)
        torch.mps.manual_seed(args['seed'])
        device = torch.device("mps")
        # MPS does not use CUDNN, so no need for CUDNN settings
        # Use deterministic algorithms with warning for unsupported ops
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        # CUDA-specific settings for non-M1 environments
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args['seed'])
            torch.cuda.manual_seed_all(args['seed'])
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            device = torch.device("cuda")
        else:
            # Fallback to CPU if neither MPS nor CUDA is available
            device = torch.device("cpu")
            torch.use_deterministic_algorithms(True, warn_only=True)

    print(f"Using device: {device}")

    if args['model_folder'] == None:
        args['model_folder'] = './model/DyMF_2025-07-24-06_41/fold_3' 

    test_dataloader, data_dir, used_column = prepare_kfold_datasets(args)


    if args['model_type'] == 'DNRI':
        from DNRI.model import Encoder
        from DNRI.runner import evaluate
        encoder = Encoder(args)

    if args['model_type'] == 'LSTM':
        from LSTM.model import Encoder
        from LSTM.runner import evaluate
        encoder = Encoder(args)

    if args['model_type'] == 'DyMF':
        if args['use_complete_graph'] == 1:
            from DyMF.model_complete import Encoder
            from DyMF.runner import evaluate
        elif args['without_dynamic_gcn'] == 1:
            from DyMF.model_without_dynamic_gcn import Encoder
            from DyMF.runner import evaluate
        elif args['without_tactical_fusion'] == 1:
            from DyMF.model_without_tactical_fusion import Encoder
            from DyMF.runner import evaluate
        elif args['without_player_style_fusion'] == 1:
            from DyMF.model_without_player_style_fusion import Encoder
            from DyMF.runner import evaluate
        elif args['without_rally_fusion'] == 1:
            from DyMF.model_without_rally_fusion import Encoder
            from DyMF.runner import evaluate
        elif args['without_style_fusion'] == 1:
            from DyMF.model_without_style_fusion import Encoder
            from DyMF.runner import evaluate
        else:
            from DyMF.model import Encoder
            from DyMF.runner import evaluate

        encoder = Encoder(args, device)


    if args['model_type'] == 'GCN':
        if args['use_complete_graph'] == 1:
            from GCN.model_complete import Encoder
        else:
            from GCN.model import Encoder
        from GCN.runner import evaluate
        encoder = Encoder(args)

    if args['model_type'] == 'ShuttleNet':
        from ShuttleNet.ShuttleNet import ShotGenEncoder, ShotGenPredictor
        from ShuttleNet.runner import evaluate
        encoder = ShotGenEncoder(args)

    if args['model_type'] == 'rGCN':
        if args['use_complete_graph'] == 1:
            from rGCN.model_complete import Encoder
        else:
            from rGCN.model import Encoder
        from rGCN.runner import evaluate
        encoder = Encoder(args, device)
    

    if args['model_type'] == 'Transformer':
        from Transformer.transformer import TransformerEncoder, TransformerPredictor
        from Transformer.runner import evaluate
        encoder = TransformerEncoder(args)

    if args['model_type'] == 'GCN_d':
        if args['use_complete_graph'] == 1:
            from GCN_dynamic.model_complete import Encoder
        else:
            from GCN_dynamic.model import Encoder
        from GCN_dynamic.runner import evaluate
        encoder = Encoder(args, device)

    if args['model_type'] == 'eGCN':
        if args['use_complete_graph'] == 1:
            from eGCN.model_complete import Encoder
        else:
            from eGCN.model import Encoder
        from eGCN.runner import evaluate
        encoder = Encoder(args)

    # encoder.load_state_dict(torch.load(args['model_folder'] + '/encoder.zip'))

    # encoder.to(device)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    ckpt_path = args['model_folder'] + '/encoder.zip'

    # 1) 先載到 CPU（最安全）
    state = torch.load(ckpt_path, map_location='cpu')

    # 2) 如果你存的是整包 dict，要取出對應 key
    # state = state['encoder']   # 視你的存檔格式而定

    encoder.load_state_dict(state)
    encoder.to(device)

    loss, auc, brier, acc = evaluate(test_dataloader, encoder, args, device=device)
    print(f"Test Loss: {loss:.4f}, Test Acc: {acc:.4f}, Test AUC: {auc:.4f}, Test Brier: {brier:.4f}")

if __name__ == "__main__":
    main()