"""在 data/dataset.json 中随机将约 1/3 的 (sign==1 and label==1) 的样本的 label 置为 0，
并把结果保存为同目录的 data_dropout.json。

用法：python ai_peigui/preprocess/process6.py [--seed 42]
"""

import os
import json
import random
import argparse
import math


def drop_labels(dataset_path, out_path, seed=None):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 收集符合条件的索引
    candidate_idxs = [i for i, item in enumerate(data) if item.get('sign') == 1 and item.get('label') == 1]
    total_candidates = len(candidate_idxs)

    if total_candidates == 0:
        print('没有找到符合 (sign==1 and label==1) 的样本，未做任何修改。')
        # 仍然写出一个副本
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return {'total': len(data), 'candidates': 0, 'dropped': 0}

    # 计算要置零的数量：大致 1/3；若样本 < 3，则至少置1
    if total_candidates < 3:
        k = 1
    else:
        k = max(1, total_candidates // 3)

    rng = random.Random(seed)
    to_drop = rng.sample(candidate_idxs, k)

    for idx in to_drop:
        data[idx]['label'] = 0

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    return {'total': len(data), 'candidates': total_candidates, 'dropped': len(to_drop), 'dropped_idxs': to_drop}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None, help='随机种子（可选），以便复现')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(base_dir, '..', 'data'))
    dataset_path = os.path.join(data_dir, 'dataset.json')
    out_path = os.path.join(data_dir, 'data_dropout.json')

    if not os.path.exists(dataset_path):
        print(f'未找到 dataset.json: {dataset_path}')
        return

    stats = drop_labels(dataset_path, out_path, seed=args.seed)
    print('完成：')
    print(f"  总样本数: {stats.get('total')}")
    print(f"  符合(sign==1,label==1) 的候选数: {stats.get('candidates')}")
    print(f"  被置为0的数量: {stats.get('dropped')}")
    if isinstance(stats.get('dropped_idxs'), (list, tuple)):
        print(f"  示例被修改的索引(前20): {stats['dropped_idxs'][:20]}")
    print(f"  输出文件: {out_path}")


if __name__ == '__main__':
    main()
