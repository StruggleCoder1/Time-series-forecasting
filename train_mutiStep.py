from joblib import dump, load
import torch.utils.data as Data
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import time
import matplotlib.pyplot as plt
import torch.nn.init as init
from kan import KAN
# 参数与配置
torch.manual_seed(100)  # 设置随机种子，以使实验结果具有可重复性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据集
def dataloader(batch_size, workers=2):
    # 训练集
    train_set = load('datasetProcess/VMD_process/train_set')
    train_label = load('datasetProcess/VMD_process/train_label')
    # 测试集
    test_set = load('datasetProcess/VMD_process/test_set')
    test_label = load('datasetProcess/VMD_process/test_label')
    # 加载数据
    train_loader = Data.DataLoader(dataset=Data.TensorDataset(train_set, train_label),
                                   batch_size=batch_size, num_workers=workers, drop_last=True)
    test_loader = Data.DataLoader(dataset=Data.TensorDataset(test_set, test_label),
                                  batch_size=batch_size, num_workers=workers, drop_last=True)
    return train_loader, test_loader

batch_size = 128
# 加载数据
train_loader, test_loader = dataloader(batch_size)
class BiLSTMChnnel(nn.Module):
    def __init__(self, n_inputs, n_outputs, num_layers=1, dropout=0.3, bidirectional=True):
        super(BiLSTMChnnel, self).__init__()
        # self.kan = KAN(width=[n_outputs * 2, n_outputs])
        self.lstm = nn.LSTM(
            input_size=n_inputs,
            hidden_size=n_outputs  ,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        # 全连接层维度需要调整
        self.fc = nn.Linear(n_outputs * 2 , n_outputs)
        self.relu = nn.ReLU()
        # 归一化层大小也需调整
        self.layer_norm = nn.LayerNorm(n_outputs)
        # 调整残差映射的大小
        self.residual_mapping = nn.Linear(n_inputs, n_outputs )
        self.batch_norm = nn.BatchNorm1d(n_outputs)  # 添加Batch Normalization
        # 初始化
        self.init_weights()
    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:  # 输入权重
                init.kaiming_normal_(param.data, mode='fan_in', nonlinearity='linear')
            elif 'weight_hh' in name:  # 隐藏权重
                init.kaiming_normal_(param.data, mode='fan_in', nonlinearity='linear')
            elif 'bias' in name:  # 偏置
                init.constant_(param.data, 0)

        # 对全连接层使用Xavier初始化
        init.xavier_normal_(self.fc.weight)
        init.constant_(self.fc.bias, 0)

    # 前向传播过程
    def forward(self, x):
        # 前向传播的过程中，identity变量也需要经过self.residual_mapping来调整大小
        identity = self.residual_mapping(x)

        # LSTM层
        out, (hidden, cell) = self.lstm(x)
        out = self.dropout(out)
        # 全连接层
        out = self.fc(out)
        out = self.batch_norm(out.permute(0, 2, 1)).permute(0, 2, 1)  # Batch Normalization
        # 激活函数
        out = self.relu(out)
        # 归一化
        out = self.layer_norm(out)
        # 添加残差连接
        out += identity
        return out

class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim):
        super(CrossAttention, self).__init__()
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim

        self.query_linear = nn.Linear(query_dim, query_dim)
        self.key_linear = nn.Linear(key_dim, query_dim)
        self.value_linear = nn.Linear(value_dim, query_dim)

        self.norm_layer = nn.LayerNorm(query_dim)

    def forward(self, query, key, value):
        # 计算查询序列的注意力权重
        query_emb = self.query_linear(query)
        key_emb = self.key_linear(key)
        attention_weights = torch.bmm(query_emb, key_emb.transpose(1, 2))
        attention_weights = torch.softmax(attention_weights, dim=-1)
        value_emb = self.value_linear(value)
        attended_values = torch.bmm(attention_weights, value_emb)
        return attended_values
def sinusoidal_positional_encoding(seq_len, d_model):
        """
        正弦位置编码
        """
        position = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe
class TransformerChnnel(nn.Module):
    def __init__(self, input_dim, model_dim, n_heads, num_encoder_layers, conv_kernel_size, seq_len,use_sin_pos_encoding = False):
        super(TransformerChnnel, self).__init__()

        # 1D卷积层
        self.conv1d = nn.Conv1d(in_channels=input_dim,
                                out_channels=model_dim,
                                kernel_size=conv_kernel_size,
                                padding=(conv_kernel_size // 2))  # 保持输出序列长度与输入相同

        # 位置编码（可选）
        if use_sin_pos_encoding:
            self.positional_encoding = sinusoidal_positional_encoding(seq_len, model_dim)
        else:
            self.positional_encoding = nn.Parameter(torch.zeros(1, seq_len, model_dim))

        # Transformer的Encoder层
        encoder_layers = TransformerEncoderLayer(d_model=model_dim, nhead=n_heads,batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)

        # 分类器
        self.fc = nn.Linear(model_dim, 32)
        # LayerNorm and Dropout
        self.norm = nn.LayerNorm(model_dim)
        self.bathnorm = nn.BatchNorm1d(model_dim)
        self.dropout = nn.Dropout(dropout)
        # 初始化
        self.init_weights()
    def init_weights(self):
        # 对卷积层使用He初始化
        init.kaiming_normal_(self.conv1d.weight, mode='fan_out', nonlinearity='relu')

        # 对全连接层使用Xavier初始化
        init.xavier_normal_(self.fc.weight)

        # 初始化偏置为零
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.constant_(m.bias, 0)

    def forward(self, x):
        # 1D卷积层
        x = x.permute(0, 2, 1)
        x =self.bathnorm(self.conv1d(x))
        x = x.permute(0, 2, 1)
        # 加入位置编码（如果需要）
        x += self.positional_encoding.to(device)
        # Transformer Encoder层
        x = self.transformer_encoder(x)
        x = self.norm(x)
        x = self.dropout(x)
        out = self.fc(x)
        return out
class VBTCKNModel(nn.Module):
    def __init__(self, batch_size, input_dim, output_dim, num_channels, hidden_dim, num_layers, num_heads,pred_len,
                 dropout=0.3,):
        super(VBTCKNModel, self).__init__()
        """
        params:
        batch_size         : 批次量大小
        input_dim          : 输入数据的维度 
        output_dim         : 输出维度
        num_channels       : 多层BiLSTMChnnel变换维度
        kernel_size        : 卷积核大小
        hidden_dim         : 注意力维度
        num_layers         : Transformer编码器层数
        num_heads          : 多头注意力头数
        dropout            : drop_out比率
        pred_len           : 预测步数
        """
        # 参数
        self.batch_size = batch_size
        self.pred_len = pred_len

        # TCN 时序空间特征 参数
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            in_channels = input_dim if i == 0 else num_channels[i - 1]  # 7 32
            out_channels = num_channels[i]  # 64
            layers += [BiLSTMChnnel(in_channels, out_channels, dropout=dropout)]
        self.BiLSTMnetwork = nn.Sequential(*layers)

        # 上采样操作
        self.unsampling = nn.Conv1d(input_dim, num_channels[0], 1)

        # Transformer编码器 时序特征参数
        self.hidden_dim = hidden_dim

        self.timetransformer = TransformerChnnel(32, self.hidden_dim, num_heads, num_layers, 3, sql_len)
        # 交叉注意力模块
        self.cross_attention = CrossAttention(num_channels[0], num_channels[-1], num_channels[-1])
        # 序列平局池化
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.kan = KAN(layers_hidden=[num_channels[0], 32,output_dim * pred_len ])  # 自定义KAN的隐藏层维度


    def forward(self, input_seq):
        # 时序空间特征 提取
        tcn_input = input_seq
        tcn_features = self.BiLSTMnetwork(tcn_input)
        # 预处理  先进行上采样
        unsampling = self.unsampling(input_seq.permute(0, 2, 1))
        transformer_output = self.timetransformer(unsampling.permute(0, 2, 1))
        # 交叉注意力机制 cross_attention
        query = transformer_output
        key = tcn_features
        value = tcn_features
        cross_attention_features = self.cross_attention(query, key, value)
        # 序列平均池化操作
        output_avgpool = self.avgpool(cross_attention_features.permute(0, 2, 1))
        output_avgpool = output_avgpool.reshape(self.batch_size, -1)
        # KAN处理
        kan_output = self.kan(output_avgpool)
        predict = kan_output.view(self.batch_size, self.pred_len, -1)
        return predict


# 定义模型参数
input_dim = 14 # 输入的特征维度
sql_len = 60
pred_len = 10   
output_dim = 1  # 输出的特征维度
num_channels = [32, 64]  # 每个TemporalBlock中的输出通道数
kernel_size = 3  # 卷积核大小
dropout = 0.3# Dropout概率

# Transformer参数
hidden_dim = 64    # 注意力维度
num_layers = 2   # 编码器层数
num_heads = 2   # 多头注意力头数
model = VBTCKNModel(batch_size, input_dim, output_dim, num_channels, hidden_dim, num_layers, num_heads,pred_len, dropout=0.3)

# 定义损失函数和优化函数
model = model.to(device)
loss_function = torch.nn.HuberLoss(reduction='mean', delta=1.0)  # loss
learn_rate = 1e-3
epochs = 100
optimizer = torch.optim.AdamW(model.parameters(), learn_rate,weight_decay=1e-5)  # 优化器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# 监控测试集损失，提前停止训练
best_val_loss = float('inf')
patience = 10  # 提前停止的耐心
counter = 0
# 训练模型
def model_train(batch_size, model, epochs, loss_function, optimizer):
    # 样本长度
    train_size = len(train_loader) * batch_size
    test_size = len(test_loader) * batch_size
    # 最低MSE
    minimum_mse = 1000.
    # 最佳模型
    best_model = model
    train_mse = []  # 记录在训练集上每个epoch的 MSE 指标的变化情况   平均值
    test_mse = []  # 记录在测试集上每个epoch的 MSE 指标的变化情况   平均值
    # 计算模型运行时间
    start_time = time.time()
    for epoch in range(epochs):
        # 训练
        model.train()
        train_mse_loss = 0.
        for seq, labels in train_loader:
            seq, labels = seq.to(device), labels.to(device)
            # 每次更新参数前都梯度归零和初始化
            optimizer.zero_grad()
            # 前向传播
            y_pred = model(seq)
            loss = loss_function(y_pred, labels)
            if torch.isnan(loss):
                print(f"NaN loss detected at epoch {epoch + 1}")
                return
            train_mse_loss += loss.item()  # 计算 MSE 损失
            # 反向传播和参数更新
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        # 计算总损失
        train_av_mseloss = train_mse_loss / train_size
        train_mse.append(train_av_mseloss)

        print(f'Epoch: {epoch + 1:2} train_MSE-Loss: {train_av_mseloss:10.8f}')
        # 每一个epoch结束后，在验证集上验证实验结果。
        with torch.no_grad():
            test_mse_loss = 0.  # 保存当前epoch的MSE loss和
            for data, label in test_loader:
                data, label = data.to(device), label.to(device)
                pre = model(data)
                # 计算损失
                test_loss = loss_function(pre, label)
                test_mse_loss += test_loss.item()
            test_av_mseloss = test_mse_loss / test_size
            test_mse.append(test_av_mseloss)
            print(f'Epoch: {epoch + 1:2} test_MSE_Loss:{test_av_mseloss:10.8f}')
            # 学习率调度器根据验证集损失调整学习率
            scheduler.step(test_av_mseloss)
            # 如果当前模型的 MSE 低于于之前的最佳准确率，则更新最佳模型
            # 保存当前最优模型参数
            if test_av_mseloss < minimum_mse:
                minimum_mse = test_av_mseloss
                counter = 0  # 重置耐心计数
                best_model = model  # 更新最佳模型的参数
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping triggered")
                    break

    # 保存最好的参数
    torch.save(best_model.state_dict(), 'best_model_VBTCKN.pt')
    print(f'\nDuration: {time.time() - start_time:.0f} seconds')
    epochs = list(range(1, len(train_mse) + 1))  # epoch 的值
    loss_data_df = pd.DataFrame({
        'epoch': epochs,
        'train_MSE_loss': train_mse,
        'test_MSE_loss': test_mse
    })
    # 保存训练损失和测试损失到 CSV 文件
    loss_data_df.to_csv('training_details.csv', index=False)

    # 可视化
    plt.plot(range(len(train_mse)), train_mse, color='b', label='train_MSE-loss')
    plt.plot(range(len(test_mse)), test_mse, color='y', label='test_MSE-loss')
    plt.legend()
    plt.show()
    print(f'min_MSE: {minimum_mse}')


#  模型训练
# batch_size = 64
if __name__ == '__main__':
    model_train(batch_size, model, epochs, loss_function, optimizer)