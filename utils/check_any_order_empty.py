import json

with open(r'E:\AI\ai_cabinets-dev\output.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

num_empty = 0
for item in data:
    if 'orders' not in item or not item['orders']:
        num_empty += 1
        print(f"发现空订单项: {item}")
print(f"空订单项总数: {num_empty}")