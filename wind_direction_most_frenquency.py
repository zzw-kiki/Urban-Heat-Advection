import os
import pandas as pd
from tqdm import tqdm  # 引入 tqdm 进度条
# 将频率出现最高的主导风向添加到城市中心数据的 direction 列中
# 文件路径
frequency_dir = r"E:\UHAE\wind rose-N\2010_678_all_cities\frequency"
city_csv = r"E:\UHAE\city_center\city_center_spatial_join.csv"

# 读取城市中心数据
df_city = pd.read_csv(city_csv, encoding='utf-8')

# 构建城市名与风向的映射字典
direction_dict = {}
# 获取所有 CSV 文件名
csv_files = [f for f in os.listdir(frequency_dir) if f.endswith('.csv')]

# 遍历并加上进度条
for filename in tqdm(csv_files, desc="处理中"):
    if filename.endswith('.csv'):
        # 从文件名提取城市名
        city_name = filename.replace('2010_678_', '').replace('_风向频率玫瑰图.csv', '')

        # 读取CSV文件
        file_path = os.path.join(frequency_dir, filename)
        df = pd.read_csv(file_path, encoding='utf-8')

        # 找出最大频率
        max_freq = df['frequency'].max()

        # 找出所有最大频率对应的风向
        max_directions = df[df['frequency'] == max_freq]['wind dominant direction'].tolist()

        # 将风向保存为逗号分隔字符串
        direction_dict[city_name] = ','.join(map(str, max_directions))

# 添加 direction 列
df_city['dominant_direction_2010'] = df_city['市'].map(direction_dict)

# 保存结果
df_city.to_csv(city_csv, index=False, encoding='utf-8-sig')

print("已完成 direction 字段添加并保存。")
