import json
import numpy as np
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn
import pickle
import os


class FBASModel(nn.Module):
    def __init__(self, fbas_vocab_size, fbas_embedding_dim=8):
        super().__init__()
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

        embedded_fbas = self.fbas_embedding(fbas_indices)
        embedded_fbas_agg = embedded_fbas.mean(dim=1)

        scalar_features = torch.cat([
            time_step.view(-1, 1),
            sign.view(-1, 1),
            hour.view(-1, 1),
            day.view(-1, 1),
            month.view(-1, 1),
            day_of_week.view(-1, 1),
            is_weekend.view(-1, 1),
            fbas_count.view(-1, 1),
        ], dim=1)

        all_features = torch.cat([scalar_features, embedded_fbas_agg], dim=1)
        return self.layers(all_features)


def load_model_artifacts():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_dir = os.path.dirname(os.path.abspath(__file__))
    state_dir = os.path.join(base_dir, 'state')
    weights_dir = os.path.join(base_dir, 'weights')

    fbas_map_path = os.path.join(state_dir, 'fbas_mapping.pkl')
    scalers_path = os.path.join(state_dir, 'feature_scalers.pkl')
    if not os.path.exists(fbas_map_path):
        raise FileNotFoundError(fbas_map_path)
    if not os.path.exists(scalers_path):
        raise FileNotFoundError(scalers_path)

    with open(fbas_map_path, 'rb') as f:
        fbas_to_idx = pickle.load(f)
    with open(scalers_path, 'rb') as f:
        feature_scalers = pickle.load(f)

    model = FBASModel(fbas_vocab_size=len(fbas_to_idx), fbas_embedding_dim=8)
    model = model.to(device)

    final_model_path = os.path.join(weights_dir, 'final_model.pt')
    full_model_path = os.path.join(weights_dir, 'full_model.pt')
    if os.path.exists(final_model_path):
        state = torch.load(final_model_path, map_location=device)
        model.load_state_dict(state)
    elif os.path.exists(full_model_path):
        model = torch.load(full_model_path, map_location=device)
    else:
        raise FileNotFoundError('final_model.pt or full_model.pt under weights/')

    model.eval()
    return model, fbas_to_idx, feature_scalers, device


def preprocess_time_features(time_str):
    dt = pd.to_datetime(time_str)
    return {
        'hour': dt.hour,
        'day': dt.day,
        'month': dt.month,
        'day_of_week': dt.dayofweek,
        'is_weekend': 1 if dt.dayofweek >= 5 else 0,
    }


def preprocess_fbas_features(fbas_list, fbas_to_idx, max_fbas_count=None):
    indices = [fbas_to_idx[f] for f in fbas_list if f in fbas_to_idx]
    if max_fbas_count is None:
        max_fbas_count = max(1, len(indices))
    if len(indices) > max_fbas_count:
        indices = indices[:max_fbas_count]
    else:
        indices = indices + [0] * (max_fbas_count - len(indices))
    return {'fbas_indices': indices, 'fbas_count': len([f for f in fbas_list if f in fbas_to_idx])}


def normalize_numerical_features(features, feature_scalers):
    numerical = ['time_step', 'hour', 'day', 'month', 'day_of_week', 'fbas_count']
    out = features.copy()
    for k in numerical:
        if k in features and k in feature_scalers:
            scaler = feature_scalers[k]
            v = np.array([[features[k]]])
            out[k] = scaler.transform(v).flatten()[0]
    return out


def prepare_single_sample(sample_data, fbas_to_idx, feature_scalers, max_fbas_count=None):
    proc = {
        'time_step': sample_data.get('time_step', 0),
        'sign': sample_data.get('sign', 0),
    }
    if 'time' in sample_data:
        proc.update(preprocess_time_features(sample_data['time']))
    else:
        proc.update({'hour': 0, 'day': 1, 'month': 1, 'day_of_week': 0, 'is_weekend': 0})

    fbas_list = sample_data.get('fbas', [])
    proc.update(preprocess_fbas_features(fbas_list, fbas_to_idx, max_fbas_count))
    proc = normalize_numerical_features(proc, feature_scalers)
    return proc


def convert_to_tensor_dict_single(features, device):
    t = {}
    for k, v in features.items():
        if k == 'fbas_indices':
            arr = np.array([v])
            t[k] = torch.LongTensor(arr).to(device)
        else:
            val = v if isinstance(v, (list, np.ndarray)) else [v]
            t[k] = torch.FloatTensor(val).to(device)
    return t


def predict_single_sample(model, sample_data, fbas_to_idx, feature_scalers, device, max_fbas_count=None):
    proc = prepare_single_sample(sample_data, fbas_to_idx, feature_scalers, max_fbas_count)
    t = convert_to_tensor_dict_single(proc, device)
    model.eval()
    with torch.no_grad():
        out = model(t)
        return float(out.squeeze().item())


def predict_batch_samples(model, samples_data, fbas_to_idx, feature_scalers, device, max_fbas_count=None):
    processed = [prepare_single_sample(s, fbas_to_idx, feature_scalers, max_fbas_count) for s in samples_data]
    max_len = max(len(p['fbas_indices']) for p in processed) if processed else 1

    batch_tensors = {}
    keys = ['time_step','sign','hour','day','month','day_of_week','is_weekend','fbas_indices','fbas_count']
    for k in keys:
        if k == 'fbas_indices':
            seqs = []
            for p in processed:
                seq = list(p['fbas_indices'])
                if len(seq) < max_len:
                    seq = seq + [0] * (max_len - len(seq))
                seqs.append(seq[:max_len])
            batch_tensors[k] = torch.LongTensor(seqs).to(device)
        else:
            vals = [p[k] for p in processed]
            batch_tensors[k] = torch.FloatTensor(vals).to(device)

    model.eval()
    with torch.no_grad():
        out = model(batch_tensors)
        probs = out.squeeze().cpu().numpy()
    if len(samples_data) == 1:
        return float(probs.item())
    return probs


def load_and_predict_from_json(json_file_path, model_dir='.'):
    orig = os.getcwd()
    if model_dir != '.':
        os.chdir(model_dir)
    try:
        model, fbas_to_idx, feature_scalers, device = load_model_artifacts()
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            data = [data]
        return predict_batch_samples(model, data, fbas_to_idx, feature_scalers, device, None)
    finally:
        os.chdir(orig)


def main():
    try:
        model, fbas_to_idx, feature_scalers, device = load_model_artifacts()
        print('模型加载成功')
        sample = {"time_step":1, "sign":1, "time":"2025-07-05 23:30:00", "fbas":["ONT8","MDT4"]}
        p = predict_single_sample(model, sample, fbas_to_idx, feature_scalers, device, None)
        print('概率', p)
    except FileNotFoundError as e:
        print('缺失文件', e)


if __name__ == '__main__':
    main()
