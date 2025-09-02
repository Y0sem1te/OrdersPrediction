"""process2.5.py
从 output-FBAfilter.json 读取每个对象，按该对象的 `fbas` 字段过滤其 `orders`：
	- 如果 `fbas` 是列表，保留 orders 中 `RECEIVER` 在该列表内的记录
	- 如果 `fbas` 是 dict，则按 key 名 (仓库名) 保留 orders 中 `RECEIVER` 在 key 列表内的记录
结果写入 output-FBAfilter-delete.json（流式写入以节省内存），支持 --max 用于调试

用法：
	python ai_peigui/process2.5.py --max 200
"""
from pathlib import Path
import json
import argparse
import sys


def extract_root_list(obj):
	if isinstance(obj, list):
		return obj
	if isinstance(obj, dict):
		for v in obj.values():
			if isinstance(v, list):
				return v
		return [obj]
	return [obj]


def get_fba_names(fbas_field):
	if fbas_field is None:
		return set()
	if isinstance(fbas_field, dict):
		return set(fbas_field.keys())
	if isinstance(fbas_field, list):
		return set(fbas_field)
	# fallback: single string
	return {str(fbas_field)}


def filter_orders_by_fbas(item):
	if not isinstance(item, dict):
		return item
	orders = item.get('orders')
	fbas_field = item.get('fbas')
	fba_names = get_fba_names(fbas_field)
	if not orders or not fba_names:
		# 若没有 orders 或没有 fbas，返回 item 并把 orders 设为空列表
		item['orders'] = []
		return item

	filtered = [o for o in orders if (o.get('RECEIVER') in fba_names)]
	item['orders'] = filtered
	return item


def main():
	p = argparse.ArgumentParser()
	p.add_argument('-f', '--file', default=r"E:\AI\ai_cabinets-dev\output-FBAfilter.json", help='源文件路径')
	p.add_argument('-o', '--out', default=r"E:\AI\ai_cabinets-dev\output-FBAfilter-delete.json", help='输出文件路径')
	p.add_argument('--max', type=int, help='只处理前 N 个对象，用于调试')
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
	processed = 0
	total_in = 0
	total_out = 0

	with out.open('w', encoding='utf-8') as of:
		of.write('[\n')
		first = True
		for i, item in enumerate(items):
			if limit is not None and i >= limit:
				break
			total_in += len(item.get('orders') or [])
			new_item = filter_orders_by_fbas(item if isinstance(item, dict) else {})
			total_out += len(new_item.get('orders') or [])
			if not first:
				of.write(',\n')
			else:
				first = False
			of.write(json.dumps(new_item, ensure_ascii=False))
			processed += 1
			if processed % 500 == 0:
				print(f"已处理 {processed} 项，输入订单总数={total_in}，输出订单总数={total_out}")
		of.write('\n]\n')

	print(f"完成：处理对象={processed}, 输入订单总数={total_in}, 输出订单总数={total_out}")
	print(f"输出文件: {out.resolve()}")


if __name__ == '__main__':
	main()

