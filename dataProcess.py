import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc("font", family='Microsoft YaHei')
from vmdpy import VMD
import torch
import pandas as pd
import numpy as np
from joblib import dump
from sklearn.preprocessing import StandardScaler
# 读取数据
original_data = pd.read_csv('../../dataset/ETTm2.csv')
# 1. 检查缺失值
missing_values = original_data.isnull().sum()
# 2. 填充缺失值（只针对数值型特征）
numerical_cols = original_data.select_dtypes(include=[np.number]).columns
original_data[numerical_cols].fillna(original_data[numerical_cols].mean())

def sliding_window_interpolation(df, value_col, window_size=24, method='mean', threshold_factor=3, interp_method='linear'):
    """
    使用滑动窗口的均值或中位数替代异常值，且支持多种插值方法。
    :param df: 输入数据框，必须包含时间序列数据
    :param value_col: 需要插值的数值列名
    :param window_size: 滑动窗口大小，决定局部计算范围
    :param method: 滑动窗口统计方法：'mean' 使用均值插值，'median' 使用中位数插值
    :param threshold_factor: 异常检测的阈值因子，乘以局部标准差来确定异常
    :param interp_method: 插值方法，默认'linear'，支持'mean', 'median', 'linear', 'spline'
    :return: 处理后的数据框
    """
    df_copy = df.copy()
    values = df_copy[value_col].copy()
    # 计算局部统计量
    if method == 'mean':
        rolling_stat = values.rolling(window=window_size, min_periods=1, center=True).mean()
    elif method == 'median':
        rolling_stat = values.rolling(window=window_size, min_periods=1, center=True).median()
    else:
        raise ValueError("Method must be 'mean' or 'median'")
    
    # 使用局部标准差进行异常检测
    rolling_std = values.rolling(window=window_size, min_periods=1, center=True).std()
    threshold = threshold_factor * rolling_std
    outliers = np.abs(values - rolling_stat) > threshold
    # 标记并替换异常值
    values[outliers] = np.nan
    # 插值处理异常值
    if interp_method in ['mean', 'median']:
        values[outliers] = rolling_stat[outliers]  # 使用滚动窗口均值/中位数插值
    elif interp_method == 'linear':
        values = values.interpolate(method='linear')
    elif interp_method == 'spline':
        values = values.interpolate(method='spline', order=2)
    else:
        raise ValueError("Unsupported interpolation method. Choose from 'mean', 'median', 'linear', 'spline'.")
    # 将处理结果回填至数据框
    df_copy[value_col] = values
    return df_copy

# 使用滑动窗口插值处理
data = sliding_window_interpolation(original_data, 'OT', window_size=24, method='mean')

# 定义VMD函数
def vmd_decomposition(data):
    alpha = 2000
    tau = 0.
    K = 7
    DC = 0
    init = 1
    tol = 1e-7
    imfs, u_hat, omega = VMD(data, alpha, tau, K, DC, init, tol)

    return imfs

OT = data['OT'].values

imfs = vmd_decomposition(OT)

features = np.column_stack([imf for imf in imfs])
features_df = pd.DataFrame(features)
features_df.columns = [f'imf{i+1}' for i in range(features_df.shape[1])]
result = pd.concat([data.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)

# 选取 数值 型 变量
original_data = result.drop(columns=['date'])
# 1. 输入训练集变量
var_data = original_data
# 2. 对应y值标签为：
ylable_data = original_data[['OT']]

# 归一化处理
scaler = StandardScaler()
var_data = scaler.fit_transform(var_data)
ylable_data = scaler.fit_transform(ylable_data)  # 使用同一个 scaler

# 保存归一化模型
dump(scaler, 'scaler')

def make_data_labels(x_data: np.ndarray, y_label: np.ndarray) -> tuple:
    """
    返回 x_data: 数据集     torch.tensor
           y_label: 对应标签值  torch.tensor
    """
    x_data = torch.tensor(x_data).float()
    y_label = torch.tensor(y_label).float()
    return x_data, y_label


def data_window_maker(x_var: np.ndarray, ylable_data: np.ndarray, window_size: int, forecast_horizon: int) -> tuple:
    """
    使用滑动窗口制作数据集
    """
    data_x, data_y = [], []
    data_len = x_var.shape[0]

    for i in range(data_len - window_size - forecast_horizon + 1):
        data_x.append(x_var[i:i + window_size, :])
        data_y.append(ylable_data[i + window_size:i + window_size + forecast_horizon])

    return make_data_labels(np.array(data_x), np.array(data_y))


def make_wind_dataset(var_data: np.ndarray, ylable_data: np.ndarray, window_size: int,
                      forecast_horizon: int, split_rate: list = [0.9, 0.1]) -> tuple:
    """
    制作滑动窗口数据集
    """
    sample_len = var_data.shape[0]
    train_len = int(sample_len * split_rate[0])

    train_var = var_data[:train_len, :]
    test_var = var_data[train_len:, :]
    train_y = ylable_data[:train_len]
    test_y = ylable_data[train_len:]

    train_set, train_label = data_window_maker(train_var, train_y, window_size, forecast_horizon)
    test_set, test_label = data_window_maker(test_var, test_y, window_size, forecast_horizon)

    return train_set, train_label, test_set, test_label


# 定义滑动窗口大小和预测步长
window_size = 60
forecast_horizon = 10

# 制作数据集
train_set, train_label, test_set, test_label = make_wind_dataset(var_data, ylable_data, window_size, forecast_horizon)
# 保存数据
dump(train_set, 'train_set')
dump(train_label, 'train_label')
dump(test_set, 'test_set')
dump(test_label, 'test_label')

print('数据 形状：')
print(train_set.size(), train_label.size())
print(test_set.size(), test_label.size())
