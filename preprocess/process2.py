"""
process2.py
对 output.json 中每个对象的 `fbas` 字段进行过滤：
    - 如果对象中存在 `fbas` 且为 dict，则根据订单数量决定保留的FBA仓库数量：
      * 订单数量 < 4：保留全部FBA仓库
      * 订单数量 4-9：保留4个最高概率的FBA仓库
      * 订单数量 10-19：保留5个最高概率的FBA仓库  
      * 订单数量 >= 20：保留6个最高概率的FBA仓库
    - 将原来的 `fbas` 字段替换为仅包含保留 fba 名称的数组（不保留概率）
处理结果写入同目录下的 `output-FBAfilter.json`（可通过参数覆盖）。

用法示例：
    python ai_peigui/process2.py -f E:/AI/ai_cabinets-dev/output.json -o E:/AI/ai_cabinets-dev/output-FBAfilter.json
"""
from pathlib import Path
import json
import argparse
import sys
from math import floor


def extract_root_list(obj):
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        # 如果 dict 中包含一个列表值则返回该列表（常见包装），否则把 dict 当作单元素列表
        for v in obj.values():
            if isinstance(v, list):
                return v
        return [obj]
    return [obj]


def filter_fbas_in_item(item):
    # 如果不存在 fbas 或 fbas 不是 dict，直接返回原 item
    if not isinstance(item, dict):
        return item
    fbas = item.get('fbas')
    if not isinstance(fbas, dict):
        return item

    # 获取订单数量
    orders = item.get('orders', [])
    order_count = len(orders) if isinstance(orders, list) else 0
    
    # 如果订单数量少于4个，保留全部FBA仓库
    if order_count < 4:
        kept_names = list(fbas.keys())
    else:
        # 根据订单数量决定保留的FBA仓库数量（4-6个）
        if order_count < 10:
            keep = 4
        elif order_count < 20:
            keep = 5
        else:
            keep = 6
        
        # 按概率降序排序，取前keep个
        sorted_fb = sorted(fbas.items(), key=lambda kv: kv[1], reverse=True)
        kept_names = [name for name, prob in sorted_fb[:keep]]

    # 用数组替换 fbas
    item['fbas'] = kept_names
    return item


def main():
    from pathlib import Path
    base_dir = Path(__file__).resolve().parent
    default_dir = base_dir / 'transition_data'
    default_src = str(default_dir / 'output.json')
    default_out = str(default_dir / 'output-FBAfilter.json')

    p = argparse.ArgumentParser()
    p.add_argument('-f', '--file', default=default_src, help='源 JSON 文件路径')
    p.add_argument('-o', '--out', default=default_out, help='输出文件路径')
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

    # 流式写入输出文件，逐个处理 items
    out.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    kept_total = 0
    with out.open('w', encoding='utf-8') as of:
        of.write('[\n')
        first = True
        limit = args.max
        for i, item in enumerate(items):
            if limit is not None and i >= limit:
                break
            new_item = filter_fbas_in_item(item.copy() if isinstance(item, dict) else item)
            if isinstance(new_item, dict) and isinstance(new_item.get('fbas'), list):
                kept_total += len(new_item['fbas'])
            if not first:
                of.write(',\n')
            else:
                first = False
            of.write(json.dumps(new_item, ensure_ascii=False))
            processed += 1
            if processed % 500 == 0:
                print(f"已处理 {processed} 项")
        of.write('\n]\n')

    print(f"处理完成: 处理项数={processed}, 保留的 fbas 总数={kept_total}")
    print(f"输出文件: {out.resolve()}")


if __name__ == '__main__':
    main()
