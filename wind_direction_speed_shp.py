import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
import math
import os
from tqdm import tqdm

# 输入文件路径
csv_path = r"E:\UHA\city_center\city_center_spatial_join_speed_2010.csv"

# 输出文件路径
output_dir = r"E:\UHA\wind rose-N"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "speed_direction_2010_060708_all_cities.shp")

# 读取CSV
df = pd.read_csv(csv_path)

# 坐标系设置
crs_wgs84 = "EPSG:4326"

def get_utm_epsg(lon, lat):
    """根据经纬度自动选择对应的 UTM EPSG 编号"""
    zone_number = int((lon + 180) / 6) + 1
    if lat >= 0:
        return 32600 + zone_number  # 北半球
    else:
        return 32700 + zone_number  # 南半球

all_lines = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="正在处理城市"):
    x0, y0 = row['POINT_X'], row['POINT_Y']
    direction = (row['direction_2010']+180)%360  # 风向，角度
    speed = row['speed(m/s)']          # 风速，m/s
    city_name = row['市']

    # 创建起点
    start_point = Point(x0, y0)

    # 选择合适投影（单位：米）
    utm_epsg = get_utm_epsg(x0, y0)
    utm_crs = f"EPSG:{utm_epsg}"

    # 起点转换到米制坐标
    gdf_start = gpd.GeoDataFrame(geometry=[start_point], crs=crs_wgs84).to_crs(utm_crs)
    x_m, y_m = gdf_start.geometry.iloc[0].x, gdf_start.geometry.iloc[0].y

    # 计算终点（方向 + 长度）
    # 注意：direction=0 表示正北，所以 x 对应 sin，y 对应 cos
    dx = speed * math.sin(math.radians(direction)) *20000
    dy = speed * math.cos(math.radians(direction)) *20000

    end_point_m = Point(x_m + dx, y_m + dy)

    # 转回WGS84
    gdf_end = gpd.GeoDataFrame(geometry=[end_point_m], crs=utm_crs).to_crs(crs_wgs84)
    end_point = gdf_end.geometry.iloc[0]

    # 生成直线
    line = LineString([start_point, end_point])

    all_lines.append({
        "city_name": city_name,
        "direction": direction,
        "speed": speed,
        "geometry": line
    })

# 保存结果
gdf_lines = gpd.GeoDataFrame(all_lines, crs=crs_wgs84)
gdf_lines.to_file(output_path, encoding="utf-8")

print("处理完成，结果已保存到：", output_path)
