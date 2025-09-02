#!/usr/bin/env python3
"""Experiment A: embedding_dim=32, AdamW lr=1e-3, BCEWithLogitsLoss, ReduceLROnPlateau, dropout=0.2
Saves metrics and plot under ai_peigui/al_.
"""
import os
import sys
import json
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ensure repo root in path
# add workspace root (parent of ai_peigui) so `import ai_peigui` works
workspace_root = Path(__file__).resolve().parent.parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from ai_peigui.train import load_data, extract_features, normalize_features, pad_sequences, to_tensor_dict

torch.manual_seed(42)

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = BASE_DIR / 'weights'
STATE_DIR = BASE_DIR / 'state'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)


class FBASModelExpA(nn.Module):
    def __init__(self, fbas_vocab_size, fbas_embedding_dim=32):
        super().__init__()
        self.fbas_embedding = nn.Embedding(fbas_vocab_size, fbas_embedding_dim)
        input_dim = 8 + fbas_embedding_dim
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, features):
        # features contains tensors already on device
        embedded = self.fbas_embedding(features['fbas_indices'])
        embedded_agg = embedded.mean(dim=1)
        scalar = torch.cat([
            features['time_step'].view(-1,1),
            features['sign'].view(-1,1),
            features['hour'].view(-1,1),
            features['day'].view(-1,1),
            features['month'].view(-1,1),
            features['day_of_week'].view(-1,1),
            features['is_weekend'].view(-1,1),
            features['fbas_count'].view(-1,1),
        ], dim=1)
        x = torch.cat([scalar, embedded_agg], dim=1)
        return self.layers(x).squeeze(1)  # logits


def collate_batch(batch, device):
    features_batch = {k: [] for k in batch[0][0].keys()}
    labels_batch = []
    for features, label in batch:
        for k, v in features.items():
            features_batch[k].append(v)
        labels_batch.append(label)

    stacked = {}
    for k, vs in features_batch.items():
        if k == 'fbas_indices':
            stacked[k] = torch.stack([torch.tensor(v, dtype=torch.long) for v in vs]).to(device)
        else:
            stacked[k] = torch.stack([torch.tensor(v, dtype=torch.float32) for v in vs]).to(device)
    labels = torch.stack([torch.tensor(l, dtype=torch.float32) for l in labels_batch]).to(device)
    return stacked, labels


def run_experiment(epochs=20, batch_size=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    data_path = BASE_DIR / 'data' / 'data_dropout.json'
    data = load_data(str(data_path))
    features, labels, fbas_to_idx = extract_features(data)
    features = normalize_features(features)

    max_fbas_count = max(len(seq) for seq in features['fbas_indices'])
    features['fbas_indices'] = pad_sequences(features['fbas_indices'], max_fbas_count)

    # split
    import numpy as np
    indices = np.arange(len(labels))
    from sklearn.model_selection import train_test_split
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

    train_feats = {k: v[train_idx] for k, v in features.items()}
    test_feats = {k: v[test_idx] for k, v in features.items()}
    train_labels = labels[train_idx]
    test_labels = labels[test_idx]

    # dataloaders
    class DS(torch.utils.data.Dataset):
        def __init__(self, feats, labs):
            self.feats = feats
            self.labs = labs
        def __len__(self):
            return len(self.labs)
        def __getitem__(self, idx):
            sample = {k: v[idx] for k,v in self.feats.items()}
            return sample, self.labs[idx]

    train_ds = DS(train_feats, train_labels)
    test_ds = DS(test_feats, test_labels)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_batch(b, device))
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=lambda b: collate_batch(b, device))

    model = FBASModelExpA(len(fbas_to_idx), fbas_embedding_dim=32).to(device)

    # pos_weight
    pos = float((train_labels == 1).sum())
    neg = float((train_labels == 0).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32).to(device)
    print('pos, neg, pos_weight=', pos, neg, pos_weight.item())

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    metrics = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    best_val_acc = 0.0
    best_path = OUT_DIR / 'exp_A_best_model.pt'

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_feats, batch_labels in train_loader:
            optimizer.zero_grad()
            logits = model(batch_feats)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_labels.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == batch_labels).sum().item()
            total += batch_labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_feats, batch_labels in test_loader:
                logits = model(batch_feats)
                loss = criterion(logits, batch_labels)
                val_loss += loss.item() * batch_labels.size(0)
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == batch_labels).sum().item()
                val_total += batch_labels.size(0)

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        metrics['train_loss'].append(train_loss)
        metrics['val_loss'].append(val_loss)
        metrics['train_acc'].append(train_acc)
        metrics['val_acc'].append(val_acc)

        scheduler.step(val_acc)

        print(f'Epoch {epoch}/{epochs}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  val_loss={val_loss:.4f} val_acc={val_acc:.4f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), str(best_path))

    # final evaluation on test
    model.load_state_dict(torch.load(str(best_path)))
    model.eval()
    with torch.no_grad():
        test_feats_t = to_tensor_dict(test_feats, device)
        logits = model(test_feats_t)
        preds = (torch.sigmoid(logits) > 0.5).float()
        test_acc = (preds == torch.tensor(test_labels, dtype=torch.float32).to(device)).float().mean().item()

    results = {
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'metrics': metrics,
        'pos_weight': pos_weight.item()
    }

    out_json = OUT_DIR / 'exp_A_metrics.json'
    with open(out_json, 'w', encoding='utf8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # plot
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(metrics['train_loss'], label='train')
    plt.plot(metrics['val_loss'], label='val')
    plt.title('loss')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(metrics['train_acc'], label='train')
    plt.plot(metrics['val_acc'], label='val')
    plt.title('acc')
    plt.legend()
    png = OUT_DIR / 'exp_A_metrics.png'
    plt.tight_layout()
    plt.savefig(str(png))

    print('\nExperiment finished. Best val acc=%.4f, test acc=%.4f' % (best_val_acc, test_acc))
    print('Metrics JSON:', out_json)
    print('Plot:', png)


if __name__ == '__main__':
    start = time.time()
    run_experiment(epochs=20, batch_size=128)
    print('Elapsed:', time.time()-start)
