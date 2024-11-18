import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from joblib import load

from train_mutiStep import VBTCKNModel, hidden_dim, test_loader, batch_size, input_dim, output_dim, num_channels, kernel_size, num_layers, num_heads, pred_len
from matplotlib import rc
rc("font", family='Microsoft YaHei')

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the model
    model = VBTCKNModel(batch_size, input_dim, output_dim, num_channels, hidden_dim, num_layers, num_heads,pred_len, dropout=0.3)
    model.load_state_dict(torch.load('best_model_VBTCKN.pt', map_location=device))
    model.to(device)
    model.eval()
    original_data, pre_data = [], []
    scaler = load('scaler')
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            original_data.append(label.cpu())
            predictions = model(data).cpu()
            pre_data.append(predictions)

    original_data = torch.cat(original_data).numpy()
    pre_data = torch.cat(pre_data).numpy()
    original_data_reshaped = original_data.reshape(-1, output_dim)  # (11520, 1)
    original_data_inversed = scaler.inverse_transform(original_data_reshaped)
    original_data = original_data_inversed.reshape(-1, pred_len, output_dim)
    pre_data_reshaped = pre_data.reshape(-1, output_dim)  # (11520, 1)
    pre_data_inversed = scaler.inverse_transform(pre_data_reshaped)
    pre_data = pre_data_inversed.reshape(-1, pred_len, output_dim)

    plt.figure(figsize=(12, 6), dpi=100)
    plt.plot(original_data[:, 0, 0], label='原始信号', color='orange')
    plt.plot(pre_data[:, 0, 0], label='VBTCKN预测值', color='green')
    plt.title('基于变分模态分解与双通道交叉注意力网络的时间序列预测模型', fontsize=15)
    plt.legend(loc='upper left')
    plt.show()

    original_data_reshaped = original_data.reshape(-1, output_dim)  # (11520, 1)
    pre_data_reshaped = pre_data.reshape(-1, output_dim)  # (11520, 1)
    print('*' * 50)
    r2_scores, mse_scores, mae_scores ,rm_scores = [], [], [],[]

    for step in range(pred_len):
        original_step = original_data[:, step, 0].reshape(-1, 1)
        pre_step = pre_data[:, step, 0].reshape(-1, 1)
        # 分别计算每个时间步的 R2、MSE、MAE
        r2_scores.append(r2_score(original_step, pre_step))
        mse_scores.append(mean_squared_error(original_step, pre_step))
        mae_scores.append(mean_absolute_error(original_step, pre_step))
        rm_scores.append(np.sqrt(mean_squared_error(original_step, pre_step)))

    # 输出各时间步的平均结果
    print(f"R2 Score: {np.mean(r2_scores)}")
    print(f"MSE: {np.mean(mse_scores)}")
    print(f"MAE: {np.mean(mae_scores)}")
    print(f"RMSE: {np.mean(rm_scores)}")
