"""
process4.py
从 E:/AI/ai_cabinets-dev/output-FBAfilter.json 中读取根数组，对每个对象：
  - 删除 `orders` 字段（如果存在）
  - 复制两份对象，分别添加 `sign`: 0 和 1，并为每份添加 `label` 字段（空字符串）
将所有复制后的对象合并为数组写入 E:/AI/ai_cabinets-dev/preparing_dataset.json

默认直接处理全部对象。可选参数 --max 用于只处理前 N 项以便测试。

用法：
  python ai_peigui/process4.py
  python ai_peigui/process4.py --max 200
"""
from pathlib import Path
import json
import argparse
import sys
import copy


def extract_root_list(obj):
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, list):
                return v
        return [obj]
    return [obj]


def make_two_samples(item: dict):
    """删除 orders（如果存在），返回两个副本，分别带 sign=0/1 和空 label。"""
    base = copy.deepcopy(item)
    base.pop('orders', None)
    a = copy.deepcopy(base)
    b = copy.deepcopy(base)
    a['sign'] = 0
    b['sign'] = 1
    a['label'] = ""
    b['label'] = ""
    return a, b


def main():
    from pathlib import Path
    base_dir = Path(__file__).resolve().parent
    default_dir = base_dir / 'transition_data'
    default_src = str(default_dir / 'output-FBAfilter-delete.json')
    default_out = str(default_dir / 'preparing_dataset.json')

    p = argparse.ArgumentParser()
    p.add_argument('-f', '--file', default=default_src, help='源 JSON 文件（默认 output-FBAfilter-delete.json）')
    p.add_argument('-o', '--out', default=default_out, help='输出文件（默认 preparing_dataset.json）')
    p.add_argument('--max', type=int, help='可选：只处理前 N 个根对象（用于调试）')
    args = p.parse_args()

    src = Path(args.file)
    out = Path(args.out)
    if not src.exists():
        print(f"源文件不存在: {src}")
        sys.exit(2)

    try:
        with src.open('r', encoding='utf-8') as f:
            obj = json.load(f)
    except Exception as e:
        print(f"读取 JSON 失败: {e}")
        sys.exit(3)

    items = extract_root_list(obj)
    limit = args.max

    out.parent.mkdir(parents=True, exist_ok=True)
    processed_in = 0
    processed_out = 0

    with out.open('w', encoding='utf-8') as of:
        of.write('[\n')
        first = True
        for i, item in enumerate(items):
            if limit is not None and i >= limit:
                break
            a, b = make_two_samples(item if isinstance(item, dict) else {})
            for sample in (a, b):
                if not first:
                    of.write(',\n')
                else:
                    first = False
                of.write(json.dumps(sample, ensure_ascii=False))
                processed_out += 1
            processed_in += 1
            if processed_in % 500 == 0:
                print(f"已处理源对象 {processed_in} 个，输出样本 {processed_out} 个")
        of.write('\n]\n')

    print(f"完成：源对象={processed_in}, 输出样本={processed_out}")
    print(f"输出文件: {out.resolve()}")


if __name__ == '__main__':
    main()
