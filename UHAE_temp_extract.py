# 读取 thermal_lag_distance_final.xlsx，取 city, center (km), strip_1 (km), strip_2 (km) 列。
# 对于每个城市，在 regression_CPR.csv 中找到对应 city_name 的行。
# 在 regression_CPR.csv 中筛选出 distance 落在这些区间范围内的数据。
# 计算 residual_2020 - residual_2010，取平均值。
import pandas as pd
import numpy as np

# 文件路径
xlsx_path = r"E:\UHA\.."
out_path = r"E:\UHA\.."

csv_center = r"E:\UHA\regression\3km_1km_buffer_center\regression_CPR.csv"
csv_strip1 = r"E:\UHA\regression\3km_1km_buffer_strip1\regression_CPR.csv"
csv_strip2 = r"E:\UHA\regression\3km_1km_buffer_strip2\regression_CPR.csv"

# 读取数据
df_xlsx = pd.read_excel(xlsx_path, usecols=["city", "center (km)", "strip_1 (km)", "strip_2 (km)"])
df_center = pd.read_csv(csv_center)
df_strip1 = pd.read_csv(csv_strip1)
df_strip2 = pd.read_csv(csv_strip2)


# 辅助函数：解析区间字符串
def parse_intervals(interval_str):
    if pd.isna(interval_str):
        return []
    intervals = []
    for part in interval_str.split(";"):
        part = part.strip()
        if "~" in part:
            left, right = part.split("~")
            intervals.append((float(left), float(right)))
    return intervals


# 计算城市的平均差值
def calc_city_value(city, interval_str, df_csv):
    intervals = parse_intervals(interval_str)
    if not intervals:
        return np.nan

    df_city = df_csv[df_csv["city_name"] == city]
    if df_city.empty:
        return np.nan

    mask = False
    for left, right in intervals:
        mask |= (df_city["distance"] >= left) & (df_city["distance"] <= right)

    df_selected = df_city[mask]
    if df_selected.empty:
        return np.nan

    diffs = df_selected["residual_2020"] - df_selected["residual_2010"]
    return diffs.mean()


# 对每一行城市进行计算
results_center = []
results_strip1 = []
results_strip2 = []
results_mean = []

for _, row in df_xlsx.iterrows():
    city = row["city"]
    center_val = calc_city_value(city, row["center (km)"], df_center)
    strip1_val = calc_city_value(city, row["strip_1 (km)"], df_strip1)
    strip2_val = calc_city_value(city, row["strip_2 (km)"], df_strip2)

    # 分别保存
    results_center.append(center_val)
    results_strip1.append(strip1_val)
    results_strip2.append(strip2_val)

    # 计算平均值（忽略 NaN）
    vals = [v for v in [center_val, strip1_val, strip2_val] if not pd.isna(v)]
    results_mean.append(np.mean(vals) if vals else np.nan)

# 添加到 df_xlsx
df_xlsx["UHAE temp(℃) center"] = results_center
df_xlsx["UHAE temp(℃) strip1"] = results_strip1
df_xlsx["UHAE temp(℃) strip2"] = results_strip2
df_xlsx["UHAE temp(℃) all"] = results_mean  # 平均值

# 保存
df_xlsx.to_excel(out_path, index=False)


# print("✅ 已完成计算，只写入一列 thermal lag temp(℃)。")
