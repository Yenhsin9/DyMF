import subprocess
import itertools
import os
import csv
# Define hyperparameter grids
dropout_rates = [ 0.3, 0.4, 0.5]
weight_decays = [1e-6, 1e-5, 1e-4, 1e-3]
learning_rates = [0.003, 0.002, 0.004]

# Fixed arguments (modify these based on your setup)
fixed_args = [
    '--model_type', 'DyMF',  # Replace with your model type, e.g., 'DNRI', 'LSTM', etc.
    '--input_data_folder_path', './data/',  # Path to your data
    '--prepared_data_output_path', './data/dataset.csv',
    '--preprocessed_data_path', './data/dataset.csv',
    '--epochs', '20',  # Reduce epochs for faster grid search; adjust as needed
    '--train_batch_size', '32',
    '--valid_batch_size', '8',
    '--test_batch_size', '8',
    '--hidden_size', '16',
    '--player_dim', '16',
    '--type_dim', '16',
    '--location_dim', '16',
    '--num_layer', '2',
    '--num_basis', '2',
    '--k_folds', '2'
]

# Ensure results file is ready
results_file = 'grid_search_results.csv'
if not os.path.exists(results_file):
    with open(results_file, 'w', newline='') as csvfile:
        fieldnames = ['dropout', 'weight_decay', 'lr', 'best_val_loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

# Perform grid search
total_combinations = len(dropout_rates) * len(weight_decays) * len(learning_rates)
current_run = 1

for dropout, wd, lr in itertools.product(dropout_rates, weight_decays, learning_rates):
    print(f"Run {current_run}/{total_combinations}: dropout={dropout}, weight_decay={wd}, lr={lr}")
    args_list = fixed_args + [
        '--dropout', str(dropout),
        '--weight_decay', str(wd),
        '--lr', str(lr)
    ]
    try:
        result = subprocess.run(['python', 'train.py'] + args_list, check=True, capture_output=True, text=True)
        print(f"Completed: {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error in run: {e.stderr}")
    current_run += 1

print("Grid search completed. Results saved to grid_search_results.csv")