""" 给数据打上标签

输入：
  - preparing_dataset.json 位于脚本同级的 transition_data/
  - data_new.json 位于脚本同级的 label/
输出：
  - dataset.json 写入上级目录的 data/ 文件夹
"""

import json
import os


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    in_preparing = os.path.join(base_dir, 'transition_data', 'preparing_dataset.json')
    in_labels = os.path.join(base_dir, 'label', 'data_new.json')
    out_dir = os.path.abspath(os.path.join(base_dir, '..', 'data'))
    os.makedirs(out_dir, exist_ok=True)

    with open(in_labels, 'r', encoding='utf-8') as f:
        labels = json.load(f)

    with open(in_preparing, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"标签数量: {len(labels)}")
    print(f"数据条目数量: {len(data)}")

    for i in range(len(labels)):
        if i*2 + 1 >= len(data):
            break

        label = labels[i]

        if label == "1,2":
            data[i*2]['label'] = 1
            data[i*2+1]['label'] = 1
        elif label == "0":
            data[i*2]['label'] = 0
            data[i*2+1]['label'] = 0
        elif label == "1":
            data[i*2]['label'] = 1 if data[i*2].get('sign') == 0 else 0
            data[i*2+1]['label'] = 1 if data[i*2+1].get('sign') == 0 else 0
        elif label == "2":
            data[i*2]['label'] = 1 if data[i*2].get('sign') == 1 else 0
            data[i*2+1]['label'] = 1 if data[i*2+1].get('sign') == 1 else 0
        else:
            print(f"警告: 未知标签 '{label}' 在索引 {i}")
            data[i*2]['label'] = 0
            data[i*2+1]['label'] = 0

    label_counts = {}
    for item in data:
        lab = item.get('label')
        label_counts[lab] = label_counts.get(lab, 0) + 1

    print(f"标签分布: {label_counts}")

    out_path = os.path.join(out_dir, 'dataset.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()