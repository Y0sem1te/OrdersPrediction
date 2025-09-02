"""
process3.py
将 `FBAfilter-preview.json` 中每个根对象的 `orders` 提取出来，放入 `calculation.json` 模板的 `ladings[*].orders` 中，
为每个输入对象生成一个完整的 calculation JSON 对象，最终把所有 calculation 对象封装成数组写入 `costs.json`。

用法示例：
  python ai_peigui/process3.py -i E:/AI/ai_cabinets-dev/FBAfilter-preview.json -t E:/AI/ai_cabinets-dev/calculation.json -o E:/AI/ai_cabinets-dev/costs.json --max 200
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


def inject_orders_into_template(template: dict, orders: list) -> dict:
	"""
	深拷贝模板并只替换模板中第一个 lading 的 `details` 字段为传入的 orders。
	不会修改模板中其它字段（比如 orderNumbers）。
	如果模板没有 ladings，则在根级创建 `details` 字段以兼容。
	"""
	t = copy.deepcopy(template)
	if isinstance(t.get('ladings'), list) and len(t['ladings']) > 0:
		t['ladings'][0]['details'] = orders
	else:
		t['details'] = orders

	return t


# 内嵌的 calculation 模板（来自 calculation.json），脚本将复制此结构并仅替换其中的 ladings[0].details
TEMPLATE = {
	"cabiningNumber": "37839a3d9aa94c95b1113811656ef6c5",
	"isPrivate": 0,
	"ladings": [
		{
			"cabiningNumber": None,
			"shippingCode": "COSCO（NY）",
			"containerType": 1,
			"shipmentCode": "CNNGB",
			"destinationCode": "USNYC",
			"placeOfSuitcase": "CNNGB",
			"returnLocation": "CNNGB",
			"cabiningVolume": 75,
			"minCabiningVolume": 70,
			"isContinued": 0,
			"cabiningLimitWeight": None,
			"minCabiningLimitWeight": None,
			"cabinetType": None,
			"outboundOutletsCode": None,
			"shippingAgentCode": "",
			"trailerAgentCode": None,
			"brokerAgentCode": None,
			"clearanceAgentCode": None,
			"cupboardAgentCode": None,
			"overseasAgentCode": None,
			"truckAgentCode": None,
			"inboundStartTime": None,
			"inboundEndTime": None,
			"oceanFreight": None,
			"deliveryModes": None,
			"locOutletsCodes": None,
			"nationCode": "US",
			"isAllowTransfer": 1,
			"isDgCabinet": None,
			"beltFrame": None,
			"tenantCode": None,
			"ladingNumber": "AI20250826001",
			"productCodes": None,
			"fbaWarehouseCodes": None,
			"details": []
		}
	],
	"scaleVOS": [
		{
			"outletsCode": "义乌仓",
			"transferScale": 0.1
		}
	],
	"inventoryDeliverys": [],
	"stockNumberUrl": "https://ptzj8.oss-cn-hangzhou.aliyuncs.com/ai/pc/2025/08/26/dfd341d1ff6745bdbd68f15bc9734fac.json",
	"tenantCode": "30fe85c8b52f4810a71df75ffab03d40",
	"isContinued": 0,
	"isImport": 0,
	"orderNumbers": []
}


def main():
	from pathlib import Path
	base_dir = Path(__file__).resolve().parent
	default_dir = base_dir / 'transition_data'
	default_input = str(default_dir / 'output-FBAfilter-delete.json')
	default_out = str(default_dir / 'costs.json')

	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input', default=default_input, help='源 FBAfilter-preview.json 路径')
	parser.add_argument('-o', '--out', default=default_out, help='输出 costs.json 路径')
	parser.add_argument('--max', type=int, help='只处理前 N 个对象，用于调试')
	args = parser.parse_args()

	src = Path(args.input)
	out = Path(args.out)

	if not src.exists():
		print(f"源文件不存在: {src}")
		sys.exit(2)

	try:
		with src.open('r', encoding='utf-8') as f:
			src_obj = json.load(f)
	except Exception as e:
		print(f"读取源 JSON 失败: {e}")
		sys.exit(3)

	# 使用内嵌 TEMPLATE
	template = TEMPLATE
	items = extract_root_list(src_obj)

	out.parent.mkdir(parents=True, exist_ok=True)
	processed = 0
	total_orders = 0
	limit = args.max

	with out.open('w', encoding='utf-8') as of:
		of.write('[\n')
		first = True
		for i, item in enumerate(items):
			if limit is not None and i >= limit:
				break
			orders = item.get('orders') if isinstance(item, dict) else []
			calc = inject_orders_into_template(template, orders or [])
			if not first:
				of.write(',\n')
			else:
				first = False
			of.write(json.dumps(calc, ensure_ascii=False))
			processed += 1
			total_orders += len(orders or [])
			if processed % 500 == 0:
				print(f"已生成 {processed} 个 calculation")
		of.write('\n]\n')

	print(f"处理完成: 生成 calculation 个数={processed}, 总 orders={total_orders}")
	print(f"输出文件: {out.resolve()}")


if __name__ == '__main__':
	main()

