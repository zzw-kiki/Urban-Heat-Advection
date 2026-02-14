import pandas as pd
import os

# ---------------- 文件路径 ----------------
uha_path = r"E:\UHAE\UHA_result\UHA.csv"
wind_path = r"E:\UHAE\city_center\wind_speed&direction.csv"
socio_path = r"E:\UHAE\socio-economic-data\251cities_with_cluster.xlsx"
output_path = r"E:\UHAE\UHA_result\linear_analysis_UHA.csv"

# ---------------- 读取数据 ----------------
uha_df = pd.read_csv(uha_path)  # city_name 为主列
wind_df = pd.read_csv(wind_path)  # 市列用于匹配 city_name
socio_df = pd.read_excel(socio_path)  # city列用于匹配 city_name

# ---------------- 合并风速风向数据 ----------------
# 假设 wind_df 中市列名为 '市'，与 uha_df['city_name'] 匹配
wind_cols = ['市', 'dominant_direction_2020', 'direction_2020', 'speed_2020(m/s)',
             'dominant_direction_2010', 'direction_2010', 'speed_2010(m/s)',
             '风向差', '风速差', '风向差(绝对值)']
merged_df = pd.merge(uha_df, wind_df[wind_cols], left_on='city_name', right_on='市', how='left')

# 删除 wind_df 的市列（可选）
merged_df.drop(columns=['市'], inplace=True)

# ---------------- 合并社会经济数据 ----------------
socio_cols = ['city', 'GDP', 'population', 'size']
merged_df = pd.merge(merged_df, socio_df[socio_cols], left_on='city_name', right_on='city', how='left')
merged_df.drop(columns=['city'], inplace=True)

# ---------------- 保存最终结果 ----------------
os.makedirs(os.path.dirname(output_path), exist_ok=True)
merged_df.to_csv(output_path, index=False, encoding='utf-8-sig')

print("✅ 成功生成 linear_analysis_UHA.csv，路径:", output_path)
