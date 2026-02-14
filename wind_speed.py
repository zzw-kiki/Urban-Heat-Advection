'''
读取两个 CSV 文件。
按照 市 列匹配。
在风速数据里计算 总风速 = √(eastward² + northward²)。
过滤 date 在 2020 年 6、7、8 月的数据。
按城市分组求平均总风速。
将结果合并到 city_center_spatial_join.csv，添加 speed 列。
保存到指定路径。
'''
import pandas as pd
import numpy as np

# 输入文件路径
city_center_file = r"E:\UHA\city_center\city_center_spatial_join.csv"
wind_file = r"E:\UHAE\China_Daily_Prevailing_WindDir_19800101_20240801\2010_678_China_City_Prevailing_Mean_WindDir.csv"

# 输出文件路径
output_file = r"E:\UHAE\city_center\city_center_spatial_join_speed_2010.csv"

# 读取数据
df_city = pd.read_csv(city_center_file)
df_wind = pd.read_csv(wind_file)

# 确保日期列转为日期类型
df_wind["date"] = pd.to_datetime(df_wind["date"], errors="coerce")

# 计算总风速
df_wind["speed"] = np.sqrt(df_wind["eastward_wind_speed"]**2 + df_wind["northward_wind_speed"]**2)

# 筛选 2020/2010 年 6/7/8 月份数据
df_wind_678 = df_wind[(df_wind["date"].dt.year == 2010) & (df_wind["date"].dt.month.isin([6, 7, 8]))]

# 计算每个市的平均速度
df_avg_speed = df_wind_678.groupby("市")["speed"].mean().reset_index()

# 合并到 city_center_spatial_join.csv
df_result = df_city.merge(df_avg_speed, on="市", how="left")

# 保存结果
df_result.to_csv(output_file, index=False, encoding="utf-8-sig")

print("处理完成，结果已保存到：", output_file)
