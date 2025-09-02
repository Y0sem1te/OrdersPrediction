#!/usr/bin/env python3
"""Experiment C: focal loss (gamma=2) with AdamW; compute AUC and PR, save metrics and plots.
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
import numpy as np

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

# ensure workspace root is importable
workspace_root = Path(__file__).resolve().parent.parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from ai_peigui.train import load_data, extract_features, normalize_features, pad_sequences, to_tensor_dict

torch.manual_seed(42)

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent
os.makedirs(OUT_DIR, exist_ok=True)


class FBASModelC(nn.Module):
    def __init__(self, fbas_vocab_size, fbas_embedding_dim=32):
        super().__init__()
        self.fbas_embedding = nn.Embedding(fbas_vocab_size, fbas_embedding_dim)
        input_dim = 8 + fbas_embedding_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, features):
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
        return self.net(x).squeeze(1)


def focal_loss_with_logits(logits, targets, gamma=2.0, alpha=None):
    # logits: (N,), targets: (N,) float 0/1
    prob = torch.sigmoid(logits)
    targets = targets.float()
    p_t = prob * targets + (1 - prob) * (1 - targets)
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    modulator = (1.0 - p_t) ** gamma
    loss = modulator * bce
    if alpha is not None:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean()


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


def run_experiment(epochs=20, batch_size=128, gamma=2.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    data_path = BASE_DIR / 'data' / 'data_dropout.json'
    data = load_data(str(data_path))
    features, labels, fbas_to_idx = extract_features(data)
    features = normalize_features(features)

    max_fbas_count = max(len(seq) for seq in features['fbas_indices'])
    features['fbas_indices'] = pad_sequences(features['fbas_indices'], max_fbas_count)

    import numpy as np
    from sklearn.model_selection import train_test_split
    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

    train_feats = {k: v[train_idx] for k, v in features.items()}
    test_feats = {k: v[test_idx] for k, v in features.items()}
    train_labels = labels[train_idx]
    test_labels = labels[test_idx]

    class DS(torch.utils.data.Dataset):
        def __init__(self, feats, labs):
            self.feats = feats
            self.labs = labs
        def __len__(self):
            return len(self.labs)
        def __getitem__(self, idx):
            sample = {k: v[idx] for k,v in self.feats.items()}
            return sample, self.labs[idx]

    train_loader = torch.utils.data.DataLoader(DS(train_feats, train_labels), batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_batch(b, device))
    test_loader = torch.utils.data.DataLoader(DS(test_feats, test_labels), batch_size=batch_size, shuffle=False, collate_fn=lambda b: collate_batch(b, device))

    model = FBASModelC(len(fbas_to_idx), fbas_embedding_dim=32).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)

    metrics = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_auc = 0.0
    best_path = OUT_DIR / 'exp_C_best_model.pt'

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_feats, batch_labels in train_loader:
            optimizer.zero_grad()
            logits = model(batch_feats)
            loss = focal_loss_with_logits(logits, batch_labels, gamma=gamma)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_labels.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == batch_labels).sum().item()
            total += batch_labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # validation metrics
        model.eval()
        val_losses = []
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for batch_feats, batch_labels in test_loader:
                logits = model(batch_feats)
                loss = focal_loss_with_logits(logits, batch_labels, gamma=gamma)
                val_losses.append(loss.item() * batch_labels.size(0))
                probs = torch.sigmoid(logits).cpu().numpy()
                val_preds.extend(probs.tolist())
                val_targets.extend(batch_labels.cpu().numpy().tolist())

        val_loss = sum(val_losses) / len(test_labels)
        val_auc = roc_auc_score(val_targets, val_preds)
        precision, recall, _ = precision_recall_curve(val_targets, val_preds)
        pr_auc = auc(recall, precision)
        val_pred_labels = (np.array(val_preds) > 0.5).astype(float)
        val_acc = (val_pred_labels == np.array(val_targets)).mean()

        metrics['train_loss'].append(train_loss)
        metrics['val_loss'].append(val_loss)
        metrics['train_acc'].append(train_acc)
        metrics['val_acc'].append(val_acc)

        print(f'Epoch {epoch}/{epochs}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_auc={val_auc:.4f} pr_auc={pr_auc:.4f}')

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), str(best_path))

    # final eval
    model.load_state_dict(torch.load(str(best_path)))
    model.eval()
    with torch.no_grad():
        test_feats_t = to_tensor_dict(test_feats, device)
        logits = model(test_feats_t)
        probs = torch.sigmoid(logits).cpu().numpy()
        auc_score = roc_auc_score(test_labels, probs)
        precision, recall, _ = precision_recall_curve(test_labels, probs)
        pr_auc_score = auc(recall, precision)
        pred_labels = (probs > 0.5).astype(float)
        test_acc = (pred_labels == test_labels).mean()

    results = {
        'best_val_auc': best_val_auc,
        'final_test_auc': auc_score,
        'final_pr_auc': pr_auc_score,
        'test_acc': float(test_acc),
        'metrics': metrics
    }

    out_json = OUT_DIR / 'exp_C_metrics.json'
    with open(out_json, 'w', encoding='utf8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # plot PR and ROC
    plt.figure(figsize=(10,5))
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
    png = OUT_DIR / 'exp_C_metrics.png'
    plt.tight_layout()
    plt.savefig(str(png))

    # also save ROC/PR data
    roc_pr_json = OUT_DIR / 'exp_C_roc_pr.json'
    with open(roc_pr_json, 'w', encoding='utf8') as f:
        json.dump({'final_test_auc': float(auc_score), 'final_pr_auc': float(pr_auc_score)}, f, indent=2)

    print('\nExperiment C finished. final_test_auc=%.4f final_pr_auc=%.4f test_acc=%.4f' % (auc_score, pr_auc_score, test_acc))
    print('Metrics JSON:', out_json)
    print('Plot:', png)


if __name__ == '__main__':
    start = time.time()
    run_experiment(epochs=20, batch_size=128, gamma=2.0)
    print('Elapsed:', time.time()-start)
