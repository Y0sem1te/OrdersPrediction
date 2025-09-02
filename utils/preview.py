import argparse
import json
from pathlib import Path
import sys


def extract_items(obj):
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, list):
                return v
        return [obj]
    return [obj]


DEFAULT_MAPPING = {
    'output': 'output.json',
    'fba': 'output-FBAfilter.json',
    'delete': 'output-FBAfilter-delete.json',
    'costs': 'new_costs.json',
}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Generate JSON preview from transition_data')
    parser.add_argument('--mode', choices=list(DEFAULT_MAPPING.keys()), default='costs', help='Which default source to use')
    parser.add_argument('--src', type=str, help='Explicit source file path (overrides --mode)')
    parser.add_argument('--dst', type=str, help='Destination preview file path (optional)')
    parser.add_argument('-n', type=int, default=25, help='Number of items to include in preview')

    args = parser.parse_args(argv)

    module_dir = Path(__file__).resolve().parent
    project_dir = module_dir.parent
    src_base_candidates = [project_dir / 'transition_data', project_dir / 'preprocess' / 'transition_data']
    src_base = None
    for c in src_base_candidates:
        if c.exists():
            src_base = c
            break
    if src_base is None:
        src_base = src_base_candidates[0]
    dst_base = project_dir / 'preprocess' / 'preview'
    dst_base.mkdir(parents=True, exist_ok=True)

    if args.src:
        src_path = Path(args.src)
    else:
        src_filename = DEFAULT_MAPPING.get(args.mode, 'new_costs.json')
        src_path = src_base / src_filename

    print(f"使用源目录: {src_base}")

    if args.dst:
        dst_path = Path(args.dst)
    else:
        dst_path = dst_base / f"{src_path.stem}-preview.json"

    if not src_path.exists():
        print(f"源文件不存在: {src_path}")
        return 1

    try:
        with src_path.open('r', encoding='utf-8') as f:
            obj = json.load(f)
    except Exception as e:
        print(f"读取 JSON 失败: {e}")
        return 2

    items = extract_items(obj)
    total_objects = len(items)
    orders_counts = []
    for it in items:
        if isinstance(it, dict) and 'orders' in it and isinstance(it['orders'], list):
            orders_counts.append(len(it['orders']))
        else:
            cnt = 0
            if isinstance(it, dict):
                for v in it.values():
                    if isinstance(v, list):
                        cnt = len(v)
                        break
            orders_counts.append(cnt)
    total_orders = sum(orders_counts)
    avg_orders = (total_orders / total_objects) if total_objects > 0 else 0

    print(f"文件: {src_path.resolve()}")
    print(f"根数组对象总数: {total_objects}")
    print(f"所有对象中的 orders 总数: {total_orders}")
    print(f"平均每个对象的 orders 数: {avg_orders:.4f}\n")

    preview = items[: args.n]

    try:
        with dst_path.open('w', encoding='utf-8') as f:
            json.dump(preview, f, ensure_ascii=False, indent=2)
        print(f"已保存前 {len(preview)} 项到: {dst_path}")
    except Exception as e:
        print(f"保存预览失败: {e}")
        return 3

    return 0


if __name__ == "__main__":
    sys.exit(main())