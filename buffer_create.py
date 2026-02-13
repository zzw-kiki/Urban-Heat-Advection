import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import math
import numpy as np
import os
import tqdm
from tqdm import tqdm
# 读取CSV
csv_path = r"E:\UHA\city_center\city_center_shifted.csv"
df = pd.read_csv(csv_path)

# 坐标系设置
crs_wgs84 = "EPSG:4326"
# crs_meter = "EPSG:3857"

# 缓存所有城市的 buffer 数据
all_buffers = []
def get_utm_epsg(lon, lat):
    """
    根据经纬度自动选择对应的 UTM EPSG 编号。
    """
    zone_number = int((lon + 180) / 6) + 1
    if lat >= 0:
        return 32600 + zone_number  # 北半球
    else:
        return 32700 + zone_number  # 南半球

for idx, row in tqdm(df.iterrows(), total=len(df), desc="正在处理城市"):

    x0, y0 = row['POINT_X_5km_2'], row['POINT_Y_5km_2']
    direction = row['direction']
    city_name = row['市']
    # tqdm.write(f"→ 正在处理：{city_name}")
    buffer_geoms = []
    buffer_ids = []
    city_names = []

    # for i in range(-10, 11): # 一共20个点包括中心点
    for i in range(-30, 31):
        step_km = 1  # 每两个点之间相距
        dx_km = i * step_km * math.sin(math.radians(direction))  # X轴 → 经度方向
        dy_km = i * step_km * math.cos(math.radians(direction))  # Y轴 → 纬度方向

        # 粗略换算 1km ≈ 0.009°
        new_x = x0 + dx_km * 0.009
        new_y = y0 + dy_km * 0.009

        buffer_point = Point(new_x, new_y)
        buffer_geoms.append(buffer_point)
        # buffer_ids.append(10 - i)  # 正方向远端为0，反方向远端为20
        buffer_ids.append(30 - i)
        city_names.append(city_name)

    # 创建 GeoDataFrame
    gdf = gpd.GeoDataFrame({
        'city_name': city_names,
        'buffer_id': buffer_ids,
        'geometry': buffer_geoms
    }, crs=crs_wgs84)
    utm_epsg = get_utm_epsg(x0, y0)
    utm_crs = f"EPSG:{utm_epsg}"
    # 投影到米制 → buffer → 投回WGS84
    gdf_meter = gdf.to_crs(utm_crs)
    gdf_meter['geometry'] = gdf_meter.buffer(3000)  # buffer 半径
    gdf_buffer = gdf_meter.to_crs(crs_wgs84)

    # 添加到总列表
    all_buffers.append(gdf_buffer)

# 合并所有城市的buffer
merged_gdf = pd.concat(all_buffers, ignore_index=True)

# 保存合并后的大 shapefile
output_path = r"E:\UHA\buffer\China_cities_3km_buffer_1km_interval_strip_2\China_cities_3km_buffer_1km_interval_strip_2.shp"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
merged_gdf.to_file(output_path, encoding='utf-8')
