import json
import numpy as np
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import os

from tqdm import tqdm

np.random.seed(42)
torch.manual_seed(42)

base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, 'data', 'data_dropout.json')
state_dir = os.path.join(base_dir, 'state')
weights_dir = os.path.join(base_dir, 'weights')

def load_data(data_path):
    """加载并预处理JSON数据"""
    with open(data_path, 'r') as f:
        data = json.load(f)

    labeled_data = [item for item in data if item['label'] != '']
    
    if not labeled_data:
        raise ValueError("没有找到带标签的数据。请检查数据文件。")
    
    print(f"加载了 {len(labeled_data)} 条有标签的数据")
    return labeled_data

def extract_features(data):
    """提取并处理特征"""
    df = pd.DataFrame(data)
    
    df['datetime'] = pd.to_datetime(df['time'])
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    """ 获取所有唯一的fbas名称 """
    all_fbas = set()
    fba_names_path = os.path.join(base_dir, 'preprocess', 'fbas_names', 'fba-names.json')
    with open(fba_names_path, 'r', encoding='utf-8') as f:
        all_fbas = set(json.load(f))
    
    fbas_to_idx = {fba: idx for idx, fba in enumerate(sorted(all_fbas))}
    
    os.makedirs(state_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    with open(os.path.join(state_dir, 'fbas_mapping.pkl'), 'wb') as f:
        pickle.dump(fbas_to_idx, f)
    
    df['fbas_indices'] = df['fbas'].apply(lambda x: [fbas_to_idx[fba] for fba in x])
    df['fbas_count'] = df['fbas'].apply(len)
    
    features = {
        'time_step': np.array(df['time_step']),
        'sign': np.array(df['sign']),
        'hour': np.array(df['hour']),
        'day': np.array(df['day']),
        'month': np.array(df['month']),
        'day_of_week': np.array(df['day_of_week']),
        'is_weekend': np.array(df['is_weekend']),
        'fbas_indices': df['fbas_indices'].tolist(),  
        'fbas_count': np.array(df['fbas_count'])
    }
    
    labels = np.array(df['label'].astype(int))
    
    return features, labels, fbas_to_idx

def normalize_features(features, is_training=True):
    """标准化数值特征"""
    numerical_features = ['time_step', 'hour', 'day', 'month', 'day_of_week', 'fbas_count']
    scalers = {}
    
    for feature in numerical_features:
        if is_training:
            scaler = StandardScaler()
            features[feature] = scaler.fit_transform(features[feature].reshape(-1, 1)).flatten()
            scalers[feature] = scaler
        else:
            features[feature] = scalers[feature].transform(features[feature].reshape(-1, 1)).flatten()
    
    if is_training:
        os.makedirs(state_dir, exist_ok=True)
        with open(os.path.join(state_dir, 'feature_scalers.pkl'), 'wb') as f:
            pickle.dump(scalers, f)
            
    return features

class FBASModel(nn.Module):
    """PyTorch实现的FBAS预测模型（FBAS embedding 做 mean 聚合）"""
    def __init__(self, fbas_vocab_size, fbas_embedding_dim=8, max_fbas_count=10):
        super(FBASModel, self).__init__()
        self.fbas_embedding = nn.Embedding(fbas_vocab_size, fbas_embedding_dim)
        aggregated_embedding_dim = fbas_embedding_dim
        input_dim = 8 + aggregated_embedding_dim

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(0.001),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, features):
        time_step = features['time_step']
        sign = features['sign']
        hour = features['hour']
        day = features['day']
        month = features['month']
        day_of_week = features['day_of_week']
        is_weekend = features['is_weekend']
        fbas_count = features['fbas_count']
        fbas_indices = features['fbas_indices']

        # fbas_indices: (batch, seq_len) -> embedding: (batch, seq_len, embed_dim)
        embedded_fbas = self.fbas_embedding(fbas_indices)
        embedded_fbas_agg = embedded_fbas.mean(dim=1)

        """ 连接所有标量特征和聚合 embedding """
        scalar_features = torch.cat([
            time_step.view(-1, 1),
            sign.view(-1, 1),
            hour.view(-1, 1),
            day.view(-1, 1),
            month.view(-1, 1),
            day_of_week.view(-1, 1),
            is_weekend.view(-1, 1),
            fbas_count.view(-1, 1)
        ], dim=1)

        all_features = torch.cat([scalar_features, embedded_fbas_agg], dim=1)
        output = self.layers(all_features)
        return output

def pad_sequences(sequences, max_len=None):
    """填充序列到相同长度"""
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)
    
    padded_sequences = np.zeros((len(sequences), max_len), dtype=np.int32)
    for i, seq in enumerate(sequences):
        seq_len = min(len(seq), max_len)
        padded_sequences[i, :seq_len] = seq[:seq_len]
    
    return padded_sequences

def to_tensor_dict(features, device):
    """将特征字典转换为张量字典"""
    tensor_dict = {}
    for key, value in features.items():
        if key == 'fbas_indices':
            tensor_dict[key] = torch.LongTensor(value).to(device)
        else:
            tensor_dict[key] = torch.FloatTensor(value).to(device)
    return tensor_dict

def train_model(data_path, epochs=50, batch_size=128):
    """训练模型的主函数"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    data = load_data(data_path)
    features, labels, fbas_to_idx = extract_features(data)
    
    """ 标准化处理：'time_step', 'hour', 'day', 'month', 'day_of_week', 'fbas_count' """
    features = normalize_features(features)

    """ 填充FBAS序列 """
    max_fbas_count = max(len(seq) for seq in features['fbas_indices'])
    features['fbas_indices'] = pad_sequences(features['fbas_indices'], max_fbas_count)
    
    train_features = {}
    test_features = {}
    indices = np.arange(len(labels))
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
    
    for feature_name, feature_values in features.items():
        train_features[feature_name] = feature_values[train_indices]
        test_features[feature_name] = feature_values[test_indices]
    
    train_labels = labels[train_indices]
    test_labels = labels[test_indices]
    
    """转成Tensor"""
    train_features_tensor = to_tensor_dict(train_features, device)
    test_features_tensor = to_tensor_dict(test_features, device)
    
    train_labels_tensor = torch.FloatTensor(train_labels).to(device)
    test_labels_tensor = torch.FloatTensor(test_labels).to(device)
    
    class FBASDataset(torch.utils.data.Dataset):
        def __init__(self, features, labels):
            self.features = features
            self.labels = labels
            self.length = len(labels)
            
        def __len__(self):
            return self.length
            
        def __getitem__(self, idx):
            sample_features = {k: torch.tensor(v[idx], dtype=torch.float32 if k != 'fbas_indices' else torch.long) 
                              for k, v in self.features.items()}
            sample_label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return sample_features, sample_label
    
    train_dataset = FBASDataset({k: v for k, v in train_features.items()}, train_labels)
    test_dataset = FBASDataset({k: v for k, v in test_features.items()}, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             collate_fn=lambda batch: collate_batch(batch))
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            collate_fn=lambda batch: collate_batch(batch))
    
    def collate_batch(batch):
        features_batch = {k: [] for k in batch[0][0].keys()}
        labels_batch = []
        
        for features, label in batch:
            for k, v in features.items():
                features_batch[k].append(v)
            labels_batch.append(label)
        
        stacked_features = {}
        for k in features_batch.keys():
            if k == 'fbas_indices':
                stacked_features[k] = torch.stack(features_batch[k]).to(device)
            else:
                stacked_features[k] = torch.stack(features_batch[k]).to(device)
        
        labels_tensor = torch.stack(labels_batch).to(device)
        return stacked_features, labels_tensor
    
    model = FBASModel(len(fbas_to_idx), max_fbas_count=max_fbas_count).to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    os.makedirs(os.path.join(weights_dir, 'model_checkpoints'), exist_ok=True)
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        for batch_features, batch_labels in tqdm(train_loader):
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs.squeeze(), batch_labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * len(batch_labels)
            predicted = (outputs.squeeze() > 0.5).float()
            correct_predictions += (predicted == batch_labels).sum().item()
            total_samples += len(batch_labels)
        
        train_loss = running_loss / total_samples
        train_accuracy = correct_predictions / total_samples
        
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                val_outputs = model(batch_features)
                val_loss = criterion(val_outputs.squeeze(), batch_labels)
                val_running_loss += val_loss.item() * len(batch_labels)
                
                val_predicted = (val_outputs.squeeze() > 0.5).float()
                val_correct += (val_predicted == batch_labels).sum().item()
                val_total += len(batch_labels)
        
        val_loss = val_running_loss / val_total
        val_accuracy = val_correct / val_total
        
        scheduler.step(val_loss)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        print(f"Epoch {epoch+1}/{epochs}, "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(weights_dir, 'model_checkpoints', 'best_model.pt'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break  
    
    model.load_state_dict(torch.load(os.path.join(weights_dir, 'model_checkpoints', 'best_model.pt')))
    model.eval()
    with torch.no_grad():
        test_outputs = model(test_features_tensor)
        test_loss = criterion(test_outputs.squeeze(), test_labels_tensor).item()
        test_predicted = (test_outputs.squeeze() > 0.5).float()
        test_correct = (test_predicted == test_labels_tensor).sum().item()
        test_accuracy = test_correct / len(test_labels)
    
    print(f"测试集损失: {test_loss:.4f}")
    print(f"测试集准确率: {test_accuracy:.4f}")
    
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.title('模型损失')
    plt.ylabel('损失')
    plt.xlabel('Epoch')
    plt.legend(['训练集', '验证集'], loc='upper right')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies)
    plt.plot(val_accuracies)
    plt.title('模型准确率')
    plt.ylabel('准确率')
    plt.xlabel('Epoch')
    plt.legend(['训练集', '验证集'], loc='lower right')
    
    plt.tight_layout()
    show_loss_path = os.path.dirname(os.path.abspath(__file__))
    png_path = os.path.join(show_loss_path, 'training_history.png')
    plt.savefig(png_path)
    plt.close()

    torch.save(model.state_dict(), os.path.join(weights_dir, 'final_model.pt'))
    
    return model

if __name__ == "__main__":
    print("开始训练模型...")
    model = train_model(data_path)
    print("模型训练完成并已保存!")
    
    print("\n模型使用说明:")
    print("1. 使用 'final_model.pt' 加载模型权重 (推荐)")
    print("3. 使用 'fbas_mapping.pkl' 将FBAS名称转换为索引")
    print("4. 使用 'feature_scalers.pkl' 对数值特征进行标准化")
    print("5. 使用 predict.py 中的函数进行预测 (predict.py 依赖 fbas_mapping.pkl 和 feature_scalers.pkl)")
