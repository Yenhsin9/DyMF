import argparse
import torch
import numpy as np
import random
import torch.nn as nn
from datetime import datetime
import os
from prepare_dataset import prepare_dataset
from prepare_dataset import prepare_kfold_datasets
from utils import save_args_file
import csv
import optuna

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
    args.add_argument("--hidden_size", type=int, default=32)
    args.add_argument("--model_type", type=str, required=True)
    args.add_argument("--lr", type=float, default=0.0005)
    args.add_argument("--player_dim", type=int, default=16)
    args.add_argument("--type_dim", type=int, default=16)
    args.add_argument("--location_dim", type=int, default=16)
    args.add_argument("--num_layer", type=int, default=2)
    args.add_argument("--weight_decay", type=float, default=0.0001)

    args.add_argument("--epochs", type=int, default=50)
    #args.add_argument("--encode_length", type=int, required=True)
    args.add_argument("--dropout", type=float, default=0.5)

    args.add_argument("--num_basis", type=int, default=2)

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
    args.add_argument("--k_folds", type=int, default=1)
    args.add_argument("--use_optuna", type=int, default=0)

    args = args.parse_args()
    args = vars(args)

    if args['use_optuna']:
        def objective(trial):
            args['lr'] = trial.suggest_loguniform('lr', 1e-5, 1e-2)
            args['weight_decay'] = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
            args['dropout'] = trial.suggest_uniform('dropout', 0.2, 0.7)
            args['train_batch_size'] = trial.suggest_categorical('train_batch_size', [32, 64, 128])
            args['hidden_size'] = trial.suggest_categorical('hidden_size', [32, 64, 128])
            args['num_basis'] = trial.suggest_int('num_basis', 1, 5)
            
            # 重新設置 seed 與裝置
            np.random.seed(args['seed'])
            random.seed(args['seed'])
            torch.manual_seed(args['seed'])

            if torch.backends.mps.is_available():
                torch.mps.manual_seed(args['seed'])
                device = torch.device("mps")
                torch.use_deterministic_algorithms(True, warn_only=True)
            elif torch.cuda.is_available():
                torch.cuda.manual_seed(args['seed'])
                torch.cuda.manual_seed_all(args['seed'])
                os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
                torch.use_deterministic_algorithms(True, warn_only=True)

            args['device'] = device

            if args['model_folder'] is None:
                args['model_folder'] = './model/' + args['model_type'] + '_' + str(datetime.now().strftime("%Y-%m-%d-%H:%M"))

            fold_datasets, test_dataloader = prepare_kfold_datasets(args, k_folds=args['k_folds'])

            # 加載 DyMF 專用 model
            from DyMF.model import Encoder
            from DyMF.runner import train_kfold
            encoder = Encoder(args, device)

            import torch.nn as nn
            encoder_optimizer = torch.optim.Adam(encoder.parameters(),
                                                lr=args['lr'],
                                                weight_decay=args['weight_decay'])
            location_criterion = nn.MSELoss()
            shot_type_criterion = nn.CrossEntropyLoss()

            encoder.to(device), location_criterion.to(device), shot_type_criterion.to(device)

            avg_val_loss, avg_val_auc, avg_val_brier, _, _, _ = train_kfold(
                fold_datasets, test_dataloader, encoder, location_criterion,
                shot_type_criterion, encoder_optimizer, args, device=device
            )

            return avg_val_auc

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=30)

        print("\n✅ 最佳參數組合:")
        for k, v in study.best_params.items():
            print(f"{k}: {v}")
        print(f"📈 最佳驗證 AUC: {study.best_value:.4f}")
        return  # 完成 optuna 後 return 結束 main


if __name__ == "__main__":
    main()