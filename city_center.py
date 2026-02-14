import re
import pandas as pd
# 从百度地图获取城市中心点坐标，但是这份文件很久远了，2013年的，要在arcgis中做出对应修改
# 读取原始 txt 文件
file_path = r"E:\UHAE\city_center\BaiduMap_cityCenter.txt"
with open(file_path, "r", encoding="gbk") as f:
    text = f.read()

# 使用正则表达式提取城市名和坐标（经度,纬度）
pattern = r'n:"([^"]+)",g:"([\d.]+),([\d.]+)\|'

matches = re.findall(pattern, text)
matches = [(name + "市", lon, lat) for name, lon, lat in matches]
# 组织为 DataFrame
df = pd.DataFrame(matches, columns=["city_name", "longitude", "latitude"])

# 保存为 CSV
output_path = r"E:\UHAE\city_center\city_center.csv"
df.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"共提取 {len(df)} 个城市，已保存到：{output_path}")
