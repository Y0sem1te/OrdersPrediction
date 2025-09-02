import pymysql
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
        sql = "select distinct RECEIVER from `hnfsu_crossborder_ai`.T_CABINET_INVENTORY_HIS_30fe85c8b52f4810a71df75ffab03d40 where IS_FBA = 1"
        cursor.execute(sql)
        results = cursor.fetchall()
        with open(r"E:\AI\ai_cabinets-dev\fba-names.json", 'w', encoding='utf-8') as f:
            json.dump([row['RECEIVER'] for row in results], f, ensure_ascii=False)