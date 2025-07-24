import matplotlib.pyplot as plt
import os

def draw_plot(train_auc_list, val_auc_list, train_brier_list, val_brier_list, 
              train_loss_list, val_loss_list, train_acc_list, val_acc_list, 
              output_folder):
    epochs = range(1, len(train_auc_list) + 1)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_auc_list, label='Train AUC')
    plt.plot(epochs, val_auc_list, label='Val AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('AUC over Epochs')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_brier_list, label='Train Brier Score')
    plt.plot(epochs, val_brier_list, label='Val Brier Score')
    plt.xlabel('Epoch')
    plt.ylabel('Brier Score')
    plt.title('Brier Score over Epochs')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(epochs, train_loss_list, label='Train Loss')
    plt.plot(epochs, val_loss_list, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(epochs, train_acc_list, label='Train Accuracy')
    plt.plot(epochs, val_acc_list, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    plt.savefig(os.path.join(output_folder, 'metrics_plot.png'))
    plt.close()