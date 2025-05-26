import matplotlib.pyplot as plt


def draw_plot(train_auc_list,val_auc_list,train_brier_list,val_brier_list):
    epochs = range(1, len(train_auc_list) + 1)
    # 2. 畫 AUC 曲線
    plt.figure()
    plt.plot(epochs, train_auc_list, label='Train AUC')
    plt.plot(epochs, val_auc_list,   label='Val AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('AUC over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 3. 畫 Brier Score 曲線
    plt.figure()
    plt.plot(epochs, train_brier_list, label='Train Brier Score')
    plt.plot(epochs, val_brier_list,   label='Val Brier Score')
    plt.xlabel('Epoch')
    plt.ylabel('Brier Score')
    plt.title('Brier Score over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()
