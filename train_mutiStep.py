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
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.GELU(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


# InceptionBlock1D 定义
class InceptionBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7], dropout_rate=0.1):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, k, padding=k // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True)
            ) for k in kernel_sizes
        ])
        self.pool = nn.MaxPool1d(3, stride=1, padding=1)
        self.pool_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.reduce = nn.Conv1d(out_channels * (len(kernel_sizes) + 1), out_channels, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        outs = [branch(x) for branch in self.branches]
        outs.append(self.pool_conv(self.pool(x)))
        out = torch.cat(outs, dim=1)
        # out = self.reduce(out)
        out = self.dropout(out)
        return self.relu(out)

class WindowAttention(nn.Module):
    def __init__(self, model_dim, window_size, num_heads, dropout=0.05):
        super(WindowAttention, self).__init__()
        self.model_dim = model_dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(model_dim, model_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(model_dim, model_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape  # [batch_size, seq_len, model_dim]
        assert N % self.window_size == 0, "Sequence length must be divisible by window_size"
        # Reshape to [batch_size, num_windows, window_size, model_dim]
        num_windows = N // self.window_size
        x = x.view(B, num_windows, self.window_size, C)

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, num_windows, self.window_size, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(3, 0, 1, 4, 2, 5)  # [3, B, num_windows, num_heads, window_size, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention computation within windows
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(2, 3).reshape(B, num_windows, self.window_size, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Reshape back to [batch_size, seq_len, model_dim]
        x = x.view(B, N, C)
        return x

class BiLSTMChnnel(nn.Module):
    def __init__(self, n_inputs, n_outputs, num_layers=1, dropout=0.05, bidirectional=True, conv_kernel_size=3,
                 dilations=[1, 3, 5, 7]):
        super(BiLSTMChnnel, self).__init__()
        # self.kan = KAN(width=[n_outputs * 2, n_outputs])
        self.lstm = nn.LSTM(
            input_size=n_inputs,
            hidden_size=n_outputs,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        # Dilated 1D convolution
        self.conv1d = nn.Conv1d(
            in_channels=n_inputs,
            out_channels=n_inputs,  # Keep input dimension to extract local features
            kernel_size=conv_kernel_size,
            padding=(conv_kernel_size - 1) * 3 // 2,  # Adjust padding for dilated conv
            dilation=3  # Dilated convolution
        )

        self.conv_layers = InceptionBlock1D(
            in_channels=n_inputs,
            out_channels=n_inputs,
            kernel_sizes=[3, 5, 7],
            dropout_rate=dropout
        )

        self.fusion_conv = nn.Sequential(
            SEBlock(n_inputs * len(dilations), reduction=8),  # SEBlock 处理拼接后的通道
            nn.Conv1d(n_inputs * len(dilations), n_inputs, kernel_size=1)  # 降维到 n_inputs
        )
        self.conv_residual_mapping = nn.Linear(n_inputs, n_inputs)  # 卷积到LSTM的残差映射
        self.conv_norm = nn.InstanceNorm1d(n_inputs)  # 卷积后的批归一化
        self.conv_relu = nn.GELU()
        # 注意力机制
        self.dropout = nn.Dropout(dropout)
        # 全连接层维度需要调整
        self.fc = nn.Linear(n_outputs * 2, n_outputs)
        self.relu = nn.GELU()
        # 归一化层大小也需调整
        self.layer_norm = nn.LayerNorm(n_outputs)
        # 调整残差映射的大小
        self.residual_mapping = nn.Linear(n_inputs, n_outputs)
        self.batch_norm = nn.BatchNorm1d(n_outputs)  # 添加Batch Normalization
        self.lstm_norm = nn.LayerNorm(n_outputs * 2)
        self.pos_embed = nn.Parameter(torch.randn(1, 100, n_inputs))
        # 初始化
        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_hh' in name:
                nn.init.orthogonal_(param)  # 正交初始化隐藏状态权重
            if 'bias' in name:
                param.data.fill_(0.1)  # 偏置项非零初始化增强遗忘门

        # 对全连接层使用Xavier初始化
        init.xavier_normal_(self.fc.weight)
        init.constant_(self.fc.bias, 0)
        # Initialize FC and residual mappings
        init.xavier_normal_(self.residual_mapping.weight)
        init.constant_(self.residual_mapping.bias, 0)
        init.xavier_normal_(self.conv_residual_mapping.weight)
        init.constant_(self.conv_residual_mapping.bias, 0)

        # 前向传播过程

    def forward(self, x):
        # 前向传播的过程中，identity变量也需要经过self.residual_mapping来调整大小
        identity = self.residual_mapping(x)
        x_conv = x.permute(0, 2, 1)  # (batch_size, n_inputs, seq_len)

        x_conv = self.conv_layers(x_conv)
        x_conv = self.fusion_conv(x_conv)  # [batch_size, n_inputs, seq_len]
        x_conv = self.conv_norm(x_conv)  # 批归一化
        x_conv = x_conv + self.pos_embed[:, :x_conv.size(2), :].transpose(1, 2)  # [batch, n_inputs, seq_len]
        x_conv = self.conv_relu(x_conv)  # 激活
        x_conv = x_conv.permute(0, 2, 1)  # 恢复 (batch_size, seq_len, n_inputs)
        # LSTM层
        out, (hidden, cell) = self.lstm(x_conv)
        out = self.lstm_norm(out)  # 在序列维度归一化
        # out = self.dropout(out)
        # 全连接层
        out = self.fc(out)
        # 激活函数
        out = self.relu(out)
        # 归一化
        out = self.layer_norm(out)
        # 添加残差连接
        out += identity
        return out

class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, d_model=32, num_heads=1, dropout=0.05):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"

        # 合并 QKV 投影，减少参数
        self.qkv_proj = nn.Linear(query_dim, d_model * 3)
        self.qkv_proj_alt = nn.Linear(key_dim, d_model * 3)

        # 增强门控机制
        self.gate_linear = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 2),  # 使用 attended, attended_alt 拼接
            nn.GELU(),
            nn.Linear(d_model // 2, 2),
            nn.Softmax(dim=-1)
        )

        # 归一化和正则化
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)  # 注意力权重 dropout
        self.residual = nn.Linear(query_dim, d_model)

    def forward(self, query, key, value):
        batch, seq_len, _ = query.size()
        # 输入清洗
        query = torch.nan_to_num(query, nan=0.0)
        key = torch.nan_to_num(key, nan=0.0)
        value = torch.nan_to_num(value, nan=0.0)

        # 正向分支 QKV 投影
        qkv = self.qkv_proj(query).view(batch, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # [batch, seq_len, num_heads, head_dim]
        q = q.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 使用高效注意力
        attn_output = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_dropout.p if self.training else 0.0
        )
        attended = self.norm(attn_output.transpose(1, 2).reshape(batch, seq_len, self.d_model))

        # 反向分支
        qkv_alt = self.qkv_proj_alt(key).view(batch, seq_len, 3, self.num_heads, self.head_dim)
        q_alt, k_alt, v_alt = qkv_alt[:, :, 0], qkv_alt[:, :, 1], qkv_alt[:, :, 2]
        q_alt = q_alt.transpose(1, 2)
        k_alt = k_alt.transpose(1, 2)
        v_alt = v_alt.transpose(1, 2)
        attn_output_alt = F.scaled_dot_product_attention(
            q_alt, k_alt, v_alt, dropout_p=self.attn_dropout.p if self.training else 0.0
        )
        attended_alt = self.norm(attn_output_alt.transpose(1, 2).reshape(batch, seq_len, self.d_model))

        # 增强门控融合
        gate_input = torch.cat([attended, attended_alt], dim=-1)  # [batch, seq_len, 3 * d_model]
        gates = self.gate_linear(gate_input)  # [batch, seq_len, 2]
        gate1, gate2 = gates.unbind(dim=-1)
        output = gate1.unsqueeze(-1) * attended + gate2.unsqueeze(-1) * attended_alt

        # 残差连接
        residual = self.residual(query)
        output = output + residual
        output = self.norm(output)
        return output
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
    def __init__(self, input_dim, model_dim, n_heads, num_encoder_layers, conv_kernel_size, seq_len, dropout,
                 dilations=[1, 3, 5, 7], use_sin_pos_encoding=True, window_size=6):
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
        self.window_size = self.get_dynamic_window_size(seq_len, ratio=0.1)  # 例如，60 * 0.1 = 6
        # Transformer的Encoder层
        encoder_layers = TransformerEncoderLayer(d_model=model_dim, nhead=n_heads, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)
        self.encoder_layers = nn.ModuleList([
            nn.ModuleDict({
                # 'global_attn': LinformerAttention(model_dim, n_heads,sql_len, dropout,proj_dim=32),  # Global sparse attention
                'global_attn': nn.MultiheadAttention(embed_dim=model_dim, num_heads=n_heads, dropout=dropout,
                                                     batch_first=True),  # Global sparse attention
                'local_attn': WindowAttention(model_dim, self.window_size, n_heads, dropout),  # Local window attention
                'ffn': nn.Sequential(
                    nn.Linear(model_dim, model_dim * 4),
                    nn.GELU(),
                    # nn.Dropout(dropout),
                    # nn.Linear(model_dim * 8, model_dim *4),
                    # nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(model_dim * 4, model_dim)
                ),
                'norm1': nn.LayerNorm(model_dim),
                'norm2': nn.LayerNorm(model_dim),
                'norm3': nn.LayerNorm(model_dim)
            }) for _ in range(num_encoder_layers)
        ])
        self.conv_layers = InceptionBlock1D(
            in_channels=input_dim,
            out_channels=model_dim,
            kernel_sizes=[3, 5, 7],
            dropout_rate=dropout
        )
        self.channel_attention = ChannelAttention(model_dim * len(dilations))
        self.fusion_conv = nn.Sequential(
            nn.Conv1d(model_dim * len(dilations), model_dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm1d(model_dim)
        )
        # 分类器
        self.fc = nn.Linear(model_dim, 32)
        # LayerNorm and Dropout
        self.norm = nn.LayerNorm(model_dim)
        self.bathnorm = nn.BatchNorm1d(model_dim)
        self.dropout = nn.Dropout(dropout)
        # 初始化
        self.init_weights()

    def get_dynamic_window_size(self, seq_len, ratio=0.1):
        # return max(3, int(seq_len * ratio))  # 确保最小窗口大小为3
        return 6

    def init_weights(self):
        # 对卷积层使用He初始化
        init.kaiming_normal_(self.conv1d.weight, mode='fan_out', nonlinearity='relu')

        # 对全连接层使用Xavier初始化
        init.xavier_normal_(self.fc.weight)

        # 初始化偏置为零
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:  # 检查 bias 是否存在
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 1D卷积层
        x = x.permute(0, 2, 1)  # 调整维度 [batch_size, input_dim, seq_len]
        x_conv = self.conv_layers(x)
        x_conv = self.channel_attention(x_conv)
        x_conv = self.fusion_conv(x_conv)  # [batch_size, n_inputs, seq_len]
        x_conv = self.bathnorm(x_conv)
        x = x_conv.permute(0, 2, 1)  # 调整维度回 [batch_size, seq_len, model_dim]

        # 加入位置编码（如果需要）
        x += self.positional_encoding.to(device)
        # Transformer Encoder with Hybrid Attention
        for layer in self.encoder_layers:
            residual = x
            # Global attention (ProbSparseAttention)
            # x = layer['global_attn'](x)
            x, _ = layer['global_attn'](query=x, key=x, value=x)
            x = layer['norm1'](x + residual)

            residual = x
            # Local window attention
            x = layer['local_attn'](x)
            x = layer['norm2'](x + residual)

            residual = x
            # Feed-forward network
            x = layer['ffn'](x)
            x = layer['norm3'](x + residual)
        x = self.norm(x)
        x = self.dropout(x)
        out = self.fc(x)

        return out
class FeatureAttention(nn.Module):
    def __init__(self, feature_dim, kernel_size=3, l1_lambda=1e-4):
        super().__init__()
        self.temporal_conv = nn.Conv1d(feature_dim, feature_dim, kernel_size=kernel_size, padding=kernel_size // 2,
                                       groups=feature_dim)
        self.pool = nn.ModuleDict({
            'avg': nn.AdaptiveAvgPool1d(1),
            'max': nn.AdaptiveMaxPool1d(1)
        })
        self.attn = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )
        self.l1_lambda = l1_lambda  # L1 正则化系数

    def forward(self, x):
        # x: [B, T, F]
        x_ = x.transpose(1, 2)  # [B, F, T]
        x_ = self.temporal_conv(x_)  # [B, F, T]
        avg_pool = self.pool['avg'](x_).squeeze(-1)  # [B, F]
        max_pool = self.pool['max'](x_).squeeze(-1)  # [B, F]
        weights = self.attn(torch.cat([avg_pool, max_pool], dim=1))  # [B, F]
        # 计算 L1 正则化损失
        l1_loss = self.l1_lambda * torch.norm(weights, p=1, dim=1).mean()
        output = x * weights.unsqueeze(1)  # [B, T, F]
        return output  

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
            in_channels = num_channels[-1] if i == 0 else num_channels[i]  # 64 64
            out_channels = num_channels[-1] if i == 0 else num_channels[0]  # 64,32
            layers += [BiLSTMChnnel(in_channels, out_channels, dropout=dropout)]
        self.BiLSTMnetwork = nn.Sequential(*layers)

        # 上采样操作
        self.unsampling = nn.Conv1d(input_dim, num_channels[0], 1)
        self.unsampling_tcn = nn.Conv1d(input_dim, num_channels[1], 1)
        # Transformer编码器 时序特征参数
        self.hidden_dim = hidden_dim

        self.timetransformer = TransformerChnnel(32, self.hidden_dim, num_heads, num_layers, 3, sql_len,dropout = dropout)
        # 交叉注意力模块
        self.cross_attention = CrossAttention(num_channels[0], num_channels[0], num_channels[0])
        # 序列平局池化
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.kan = KAN(layers_hidden=[num_channels[0],output_dim * pred_len ])  # 自定义KAN的隐藏层维度

        self.feature_attn = FeatureAttention(input_dim, sql_len)

    def forward(self, input_seq):
        # 时序空间特征 提取
        # 特征重要性分析，当数据集中特征较多且复杂时使用 ,否则使用tcn_input = input_seq
        # input_seq = self.feature_attn(input_seq)
        tcn_input = self.unsampling_tcn(input_seq.permute(0, 2, 1))
        tcn_input = tcn_input.permute(0, 2, 1)
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
