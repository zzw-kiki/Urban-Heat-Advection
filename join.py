import pandas as pd
import os
import geopandas as gpd
# 按FID合并每个city的每个buffer的数据，最终输出结果中可能有缺失的FID何buffer_id，原因是arcgis pro按分区导出统计数据时，若每一层数据在该Buffer
# 内没有数据，则该Buffer的统计结果不会导出，所以最终结果中可能有缺失的FID何buffer_id，所有某一城市的某一Buffer种若有任意一层数据没有统计到，那么这条数据整条缺失
# 文件路径
path_BH_2010 = r"E:\UHAE\wind_path_statistic\3km_1km_buffer_strip_5km_2\wind_path_BH_2010.csv"
path_BH_2020 = r"E:\UHAE\wind_path_statistic\3km_1km_buffer_strip_5km_2\wind_path_BH_2020.csv"
path_MODIS_2010 = r"E:\UHAE\wind_path_statistic\3km_1km_buffer_strip_5km_2\wind_path_MODIS_2010.csv"
path_MODIS_2020 = r"E:\UHAE\wind_path_statistic\3km_1km_buffer_strip_5km_2\wind_path_MODIS_2020.csv"
path_buffer = r"E:\UHAE\buffer\China_cities_3km_buffer_1km_interval_strip_2\China_cities_3km_buffer_1km_interval_strip_2.csv"
# 读取并重命名字段
df_BH_2010 = pd.read_csv(path_BH_2010)[['FID', 'MEAN']].rename(columns={'MEAN': 'BH_2010'})
df_BH_2020 = pd.read_csv(path_BH_2020)[['FID', 'MEAN']].rename(columns={'MEAN': 'BH_2020'})
df_MODIS_2010 = pd.read_csv(path_MODIS_2010)[['FID', 'MEAN']].rename(columns={'MEAN': 'MODIS_2010'})
df_MODIS_2020 = pd.read_csv(path_MODIS_2020)[['FID', 'MEAN']].rename(columns={'MEAN': 'MODIS_2020'})
df_city_name = pd.read_csv(path_buffer)[['FID', 'city_name']]
df_buffer_id = pd.read_csv(path_buffer)[['FID', 'buffer_id']]
# ×0.1 处理 MODIS 值
df_MODIS_2010['MODIS_2010'] *= 0.1
df_MODIS_2020['MODIS_2020'] *= 0.1

# 按 FID 合并
df_merged = df_BH_2010.merge(df_BH_2020, on='FID') \
                      .merge(df_MODIS_2010, on='FID') \
                      .merge(df_MODIS_2020, on='FID') \
                     .merge(df_city_name, on='FID') \
                     .merge(df_buffer_id, on='FID')

# 输出到目标路径
output_path = r"E:\UHAE\wind_path_statistic\3km_1km_buffer_strip_5km_2\data.csv"
df_merged.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"已成功输出到：{output_path}")