#!/usr/bin/env python3
"""Compare baseline final_model.pt with experiment D best model on the same test split.
Writes JSON report to ai_peigui/al_/compare_report.json and prints a summary.
"""
import sys
from pathlib import Path
import json
import os

workspace_root = Path(__file__).resolve().parent.parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, accuracy_score

from ai_peigui.train import load_data, extract_features, normalize_features, pad_sequences, to_tensor_dict, FBASModel

# import experiment D model class
from ai_peigui.al_.exp_D_run import FBASModelD

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent


def evaluate_model(model, test_feats, test_labels, device):
    model.to(device)
    model.eval()
    with torch.no_grad():
        t = to_tensor_dict(test_feats, device)
        logits = model(t)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > 0.5).astype(float)
    auc_score = float(roc_auc_score(test_labels, probs))
    precision, recall, _ = precision_recall_curve(test_labels, probs)
    pr_auc = float(auc(recall, precision))
    acc = float(accuracy_score(test_labels, preds))
    return {'auc': auc_score, 'pr_auc': pr_auc, 'accuracy': acc, 'probs': probs.tolist(), 'preds': preds.tolist()}


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_path = BASE_DIR / 'data' / 'data_dropout.json'
    data = load_data(str(data_path))
    features, labels, fbas_to_idx = extract_features(data)
    features = normalize_features(features)
    max_fbas_count = max(len(seq) for seq in features['fbas_indices'])
    features['fbas_indices'] = pad_sequences(features['fbas_indices'], max_fbas_count)

    from sklearn.model_selection import train_test_split
    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

    test_feats = {k: v[test_idx] for k, v in features.items()}
    test_labels = labels[test_idx]

    # baseline model
    baseline_path = BASE_DIR / 'weights' / 'final_model.pt'
    expd_path = OUT_DIR / 'exp_D_best_model.pt'

    results = {}

    if baseline_path.exists():
        baseline = FBASModel(len(fbas_to_idx), fbas_embedding_dim=8, max_fbas_count=max_fbas_count)
        sd = torch.load(str(baseline_path), map_location='cpu')
        baseline.load_state_dict(sd)
        results['baseline'] = evaluate_model(baseline, test_feats, test_labels, device)
    else:
        results['baseline'] = {'error': 'baseline checkpoint not found', 'path': str(baseline_path)}

    if expd_path.exists():
        exp_model = FBASModelD(len(fbas_to_idx), fbas_embedding_dim=32)
        sd = torch.load(str(expd_path), map_location='cpu')
        exp_model.load_state_dict(sd)
        results['exp_d'] = evaluate_model(exp_model, test_feats, test_labels, device)
    else:
        results['exp_d'] = {'error': 'exp_d checkpoint not found', 'path': str(expd_path)}

    out = OUT_DIR / 'compare_report.json'
    with open(out, 'w', encoding='utf8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print('Comparison written to', out)
    if 'baseline' in results and 'error' not in results['baseline']:
        print('Baseline acc=%.4f auc=%.4f pr_auc=%.4f' % (results['baseline']['accuracy'], results['baseline']['auc'], results['baseline']['pr_auc']))
    if 'exp_d' in results and 'error' not in results['exp_d']:
        print('Exp D acc=%.4f auc=%.4f pr_auc=%.4f' % (results['exp_d']['accuracy'], results['exp_d']['auc'], results['exp_d']['pr_auc']))


if __name__ == '__main__':
    main()
