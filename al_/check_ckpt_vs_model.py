#!/usr/bin/env python3
"""Compare a PyTorch checkpoint state_dict with the current FBASModel state_dict.
Writes a JSON report to the same folder and prints a short summary.

Usage:
  python ai_peigui/al_/check_ckpt_vs_model.py [<checkpoint_path>]
If no checkpoint_path is given, the script will try common defaults under ai_peigui/weights/.
"""
import sys
import os
from pathlib import Path
import json

try:
    import torch
except Exception as e:
    print('ERROR: torch not available:', e)
    raise


def find_checkpoint(base_dir: Path):
    candidates = [
        base_dir / 'weights' / 'final_model.pt',
        base_dir / 'weights' / 'model_checkpoints' / 'best_model.pt',
        base_dir / 'weights' / 'full_model.pt',
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_state_dict_from_file(p: Path):
    ckpt = torch.load(str(p), map_location='cpu')
    if isinstance(ckpt, dict) and 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
        return ckpt['state_dict']
    # sometimes the checkpoint already *is* a state_dict (dict of tensors)
    if isinstance(ckpt, dict) and all(hasattr(v, 'shape') for v in ckpt.values()):
        return ckpt
    # other structures
    # try common key names
    for k in ('model_state_dict', 'model', 'net'):
        if k in ckpt and isinstance(ckpt[k], dict):
            return ckpt[k]
    raise RuntimeError('Unrecognized checkpoint format: ' + str(list(ckpt.keys())[:10]))


def main(argv):
    base_dir = Path(__file__).resolve().parent.parent
    base_dir = base_dir

    if len(argv) > 1:
        ckpt_path = Path(argv[1])
    else:
        ckpt_path = find_checkpoint(base_dir)

    if ckpt_path is None or not ckpt_path.exists():
        print('No checkpoint found. Provide a path as the first argument or place final_model.pt under ai_peigui/weights/')
        return 2

    print('Using checkpoint:', ckpt_path)
    ckpt_sd = load_state_dict_from_file(ckpt_path)

    # try to infer fbas vocab size from existing mapping
    state_dir = base_dir / 'state'
    fbas_map_path = state_dir / 'fbas_mapping.pkl'
    vocab_size = None
    if fbas_map_path.exists():
        try:
            import pickle
            with open(fbas_map_path, 'rb') as f:
                mapping = pickle.load(f)
            if isinstance(mapping, dict):
                vocab_size = len(mapping)
                print('Inferred fbas vocab size from', fbas_map_path, '=>', vocab_size)
        except Exception:
            pass

    # ensure repository root is on sys.path so `ai_peigui` can be imported
    repo_root = base_dir.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # import model class
    try:
        from ai_peigui.predict import FBASModel
    except Exception as e:
        print('ERROR: failed to import FBASModel from ai_peigui.predict:', e)
        raise

    # fallback vocab
    if vocab_size is None:
        vocab_size = 1000
        print('Falling back to default fbas_vocab_size =', vocab_size)

    # instantiate model (assume embedding_dim default as in code)
    model = FBASModel(fbas_vocab_size=vocab_size)
    model_sd = model.state_dict()

    # prepare report
    report = {
        'checkpoint_path': str(ckpt_path),
        'fbas_vocab_size_used': vocab_size,
        'ckpt_keys': {k: list(v.shape) for k, v in ckpt_sd.items()},
        'model_keys': {k: list(v.shape) for k, v in model_sd.items()},
    }

    matches = []
    mismatches = []
    missing_in_ckpt = []
    unexpected_in_ckpt = []

    for k_ck, v_ck in ckpt_sd.items():
        if k_ck in model_sd:
            if tuple(v_ck.shape) == tuple(model_sd[k_ck].shape):
                matches.append(k_ck)
            else:
                mismatches.append({'key': k_ck, 'ckpt_shape': list(v_ck.shape), 'model_shape': list(model_sd[k_ck].shape)})
        else:
            unexpected_in_ckpt.append(k_ck)

    for k in model_sd:
        if k not in ckpt_sd:
            missing_in_ckpt.append(k)

    report.update({
        'matches_count': len(matches),
        'mismatches': mismatches,
        'missing_in_ckpt': missing_in_ckpt,
        'unexpected_in_ckpt': unexpected_in_ckpt,
    })

    out_dir = Path(__file__).resolve().parent
    out_path = out_dir / 'ckpt_vs_model_report.json'
    with open(out_path, 'w', encoding='utf8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # print concise summary
    print('\nSUMMARY:')
    print('  total ckpt keys:', len(ckpt_sd))
    print('  total model keys:', len(model_sd))
    print('  matches:', report['matches_count'])
    print('  mismatches:', len(report['mismatches']))
    print('  missing_in_ckpt:', len(report['missing_in_ckpt']))
    print('  unexpected_in_ckpt:', len(report['unexpected_in_ckpt']))
    print('Detailed report written to:', out_path)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
