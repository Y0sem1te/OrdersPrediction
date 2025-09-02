"""
validate_costs.py
加载 `costs.json`（首选）或 `costs-preview.json`（备用），检查 JSON 是否为根数组，统计元素数量，
检查每个元素是否包含期望的字段（如 `ladings` 和 `ladings[0].details` 或 `details`），
并将第一个元素与 `costs-preview.json` 的第一个元素做字段对比，打印差异示例。

用法:
  python ai_peigui/validate_costs.py
"""
from pathlib import Path
import json
import sys
from typing import Any, Dict, Set


from typing import Optional


def load_json(preferred: Path, fallback: Optional[Path] = None):
    if preferred.exists():
        p = preferred
    elif fallback and fallback.exists():
        p = fallback
    else:
        print(f"找不到文件: {preferred} 或 {fallback}")
        sys.exit(2)
    try:
        with p.open('r', encoding='utf-8') as f:
            return json.load(f), p
    except Exception as e:
        print(f"读取 JSON 失败: {p} -> {e}")
        sys.exit(3)


def keys_of(obj: Any) -> Set[str]:
    if isinstance(obj, dict):
        return set(obj.keys())
    return set()


def main():
    base = Path(r"E:\AI\ai_cabinets-dev\costs.json")
    preview = Path(r"E:\AI\ai_cabinets-dev\costs-preview.json")

    data, path = load_json(base, preview)
    print(f"加载文件: {path}，类型: {type(data).__name__}")

    if not isinstance(data, list):
        print("根对象不是数组（list）——这会导致 downstream 处理失败。")
        sys.exit(4)

    n = len(data)
    print(f"根数组元素数: {n}")

    # 基本检查若干字段
    missing_ladings = 0
    missing_details = 0
    for i, item in enumerate(data[:500]):
        if not isinstance(item, dict):
            continue
        if 'ladings' not in item:
            missing_ladings += 1
        else:
            lad = item.get('ladings')
            if not (isinstance(lad, list) and len(lad) > 0 and isinstance(lad[0], dict)):
                missing_ladings += 1
            else:
                if 'details' not in lad[0]:
                    missing_details += 1

    print(f"前 500 项中缺失 ladings 的数量: {missing_ladings}")
    print(f"前 500 项中 ladings[0] 缺失 details 的数量: {missing_details}")

    # 字段差异示例：与 preview 的第一个元素比较
    if preview.exists():
        with preview.open('r', encoding='utf-8') as f:
            pv = json.load(f)
        if isinstance(pv, list) and len(pv) > 0 and len(data) > 0:
            keys_a = keys_of(data[0])
            keys_b = keys_of(pv[0])
            only_in_a = keys_a - keys_b
            only_in_b = keys_b - keys_a
            print("与 costs-preview.json 首元素键名差异样例:")
            print(f"  仅在当前文件中: {sorted(list(only_in_a))[:20]}")
            print(f"  仅在 preview 中: {sorted(list(only_in_b))[:20]}")

    print("校验完成。")


if __name__ == '__main__':
    main()
