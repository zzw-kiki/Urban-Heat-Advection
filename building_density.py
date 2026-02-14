# 统计每个城市中心内建筑物像元占比

import pandas as pd
import os

# 文件路径
dir_path = r"E:\UHAE\BH&BD"
file_2010 = os.path.join(dir_path, "building_pixel_count_2010.csv")
file_2010_all = os.path.join(dir_path, "all_pixel_count_2010.csv")
file_2020 = os.path.join(dir_path, "building_pixel_count_2020.csv")
file_2020_all = os.path.join(dir_path, "all_pixel_count_2020.csv")

# 读取数据
df_2010 = pd.read_csv(file_2010)
df_2010_all = pd.read_csv(file_2010_all)
df_2020 = pd.read_csv(file_2020)
df_2020_all = pd.read_csv(file_2020_all)

# 以 bh2020_all 为基准建立城市列表
df = df_2020_all[['市', 'COUNT']].rename(columns={'COUNT': 'COUNT_2020all'})

# 合并 2020 实际值
df = df.merge(df_2020[['市', 'COUNT']], on='市', how='left')
df = df.rename(columns={'COUNT': 'COUNT_2020'})

# 合并 2010 对照
df = df.merge(df_2010_all[['市', 'COUNT']], on='市', how='left')
df = df.rename(columns={'COUNT': 'COUNT_2010all'})

df = df.merge(df_2010[['市', 'COUNT']], on='市', how='left')
df = df.rename(columns={'COUNT': 'COUNT_2010'})

# 计算 BD
df['BD_2010'] = df['COUNT_2010'] / df['COUNT_2010all']
df['BD_2020'] = df['COUNT_2020'] / df['COUNT_2020all']

# 只保留需要的列
result = df[['市', 'BD_2010', 'BD_2020']]

# 缺失值填充为 "无"
result = result.fillna("None")

# 保存到同目录
output_file = os.path.join(dir_path, "city_BD.csv")
result.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"结果已保存至 {output_file}")

