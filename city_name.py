# 全国行政边界，风向，和城市中心的城市名需要一一对应
# 全国行政区划：2024全国行政区划(全国_市界.csv)
# 风向统计出的城市为371个（city_list_2020.txt），行政区划的城市为375个，最后统一去除三沙市（岛屿，数据缺失）
# 行政区划向风向城市统一的同时，城市中心点坐标也要向风向城市统一（city_center_spatial_join.csv）
import pandas as pd

import pandas as pd

# 读取CSV，获取“市”字段

csv_path = r"E:\UHAE\city_center\city_center_spatial_join.csv"
df = pd.read_csv(csv_path, encoding='utf-8')  # 如果报错尝试 'gbk'
csv_cities = set(df['市'].astype(str).str.strip())

# 读取txt文件，获取城市名列表
txt_path = r"E:\UHAE\wind rose-N\2020_678_all_cities\frequency\city_list_2020.txt"
with open(txt_path, 'r', encoding='utf-8') as f:
    txt_cities = set(line.strip() for line in f if line.strip())

# 查找txt中不存在于csv的城市名
missing_in_csv = txt_cities - csv_cities

if missing_in_csv:
    print("以下城市名在CSV文件中未出现：")
    for city in sorted(missing_in_csv):
        print(city)
else:
    print("txt文件中所有城市名均出现在CSV文件中。")
    # 反向查找CSV中没有出现在txt中的城市名（多余的）
    extra_in_csv = csv_cities - txt_cities
    if extra_in_csv:
        print("但CSV文件中存在以下多余的城市名（未出现在txt文件中）：")
        for city in sorted(extra_in_csv):
            print(city)
    else:
        print("且CSV文件中没有多余的城市名。")

# 风向城市中的城市全部存在于2024行政区划矢量文件中，
# 多出的四个为：中农发山丹马场————张掖市
# 太子山天然林保护区————临夏回族自治州
# 白杨市————塔城地区
# 莲花山风景林自然保护区————临夏回族自治州
# 编辑行政区划数据，对照2020行政区划，将对应地区归为市
# 运行结果应该是只有三沙市未出现在csv文件中

