# Time-series-forecasting
VBTCKN: A Time Series Prediction Model Based on Variational Mode Decomposition and Dual Channel Cross Attention Network
This model utilizes Variational Mode Decomposition (VMD) to decompose the time series into multiple modal components with different frequencies to reduce the volatility of the sequence. The BiLSTM channel is used to extract long-term dependency relationships and time-dependent features. Extracting local patterns and global dependencies through Transformer channels, and then using CrossAttention mechanism to fuse the results of dual channels to enhance the overall expression ability of the model and improve prediction accuracy. Finally, the prediction results are output through the KAN (Kolmogorov Arnold Networks) network
Operating environment：
    python=3.11
    pytorch=2.1.0
    torchvision=0.16.0
    numpy=1.26.0
    pandas=2.1.1
