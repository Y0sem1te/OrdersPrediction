import pymysql
import os
from datetime import timedelta
import datetime
import json
from tqdm import tqdm
def connect_mysql(host, user, password, database, port=3306):
    """
    连接MySQL数据库并返回连接对象
    """
    conn = pymysql.connect(
        host=host,
        user=user,
        password=password,
        database=database,
        port=port,
        charset='utf8mb4'
    )
    return conn

if __name__ == "__main__":
    conn = connect_mysql(
        host='192.168.1.40',
        user='root',
        password='tidb_20211214',
        database='ai_cabinet_test',
        port=4000
    )

    from pymysql.cursors import DictCursor
    with conn.cursor(DictCursor) as cursor:
        sql = "select * from `hnfsu_crossborder_ai`.`T_CABINET_INVENTORY_HIS_30fe85c8b52f4810a71df75ffab03d40` order by INBOUND_TIME ASC"
        cursor.execute(sql)
        results = cursor.fetchall()
        data = []
        
        start = datetime.datetime(2025, 7, 1, 0, 0, 0)
        end = datetime.datetime(2025, 7, 31, 12, 0, 0)
        current = start

        while current < end:
            for gap in range(1, 12):
                data_item = {
                    'time': current,
                    'time_step': gap,
                    'orders': [],
                    'fbas': {}
                }
                begin = current
                end_time = begin + timedelta(hours=gap)
                fba_ratio = {}
                fba_num = 0

                for row in results:
                    if row['INBOUND_TIME'] >= end_time:
                        break
                    if row['INBOUND_TIME'] >= begin:
                        if row.get('IS_FBA') == 1 or row.get('IS_FBA') == '1':
                            data_item['orders'].append(row)
                            fba_num += 1
                            recv = row.get('RECEIVER')
                            fba_ratio[recv] = fba_ratio.get(recv, 0) + 1

                for receiver, count in fba_ratio.items():
                    fba_ratio[receiver] = count / fba_num if fba_num > 0 else 0
                data_item['fbas'] = fba_ratio
                data.append(data_item)
            current += timedelta(minutes=30)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, 'transition_data')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'output.json')
    json.dump(data, open(out_path, 'w', encoding='utf-8'), default=str)
    conn.close()