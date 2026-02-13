# 合并下风口温度大于上风口的连续区间和AVG_UHAE>0的连续区间，取交集，即将两个判定准则取交集
# 先把 strip_1 和 strip_2 与 center 取并集，生成新列 center&strip1/2
# 分别计算：
# center (km) 与 lst_difference_km 的交集 → 存到 intersection_center
# center&strip1/2 与 lst_difference_km 的交集 → 存到 intersection_center&strip1/2
# 5.对两个交集列分别计算区间范围 → 存到 range_center 和 range_center&strip1/2
import pandas as pd
import numpy as np

# 解析区间字符串为区间列表
def parse_ranges(s):
    if pd.isna(s):
        return []
    ranges = []
    for part in str(s).split(';'):
        part = part.strip()
        if '~' in part:
            try:
                start, end = map(float, part.split('~'))
                ranges.append((start, end))
            except:
                continue
    return ranges

# 区间并集
def union_ranges(*list_of_ranges):
    all_ranges = []
    for ranges in list_of_ranges:
        all_ranges.extend(ranges)
    if not all_ranges:
        return []
    # 排序
    all_ranges.sort(key=lambda x: x[0])
    merged = [all_ranges[0]]
    for current in all_ranges[1:]:
        prev_start, prev_end = merged[-1]
        cur_start, cur_end = current
        if cur_start <= prev_end:  # 有重叠
            merged[-1] = (prev_start, max(prev_end, cur_end))
        else:
            merged.append(current)
    return merged

# 区间交集
def intersect_ranges(ranges1, ranges2):
    result = []
    for s1, e1 in ranges1:
        for s2, e2 in ranges2:
            start = max(s1, s2)
            end = min(e1, e2)
            if start <= end:
                result.append((round(start, 2), round(end, 2)))
    return result

# 区间范围计算：右端-左端的总和
def extract_range_sum(s):
    if pd.isna(s) or s == "":
        return np.nan
    parts = str(s).split(';')
    try:
        total_range = 0.0
        for p in parts:
            bounds = p.split('~')
            if len(bounds) == 2:
                left = float(bounds[0])
                right = float(bounds[1])
                total_range += (right - left )
        return total_range if total_range > 0 else np.nan
    except:
        return np.nan

# ===================== 主程序 =====================
input_path = r""
output_path = r""

# 读取Excel
df = pd.read_excel(input_path)

# 生成并集列（center、strip1、strip2）
unions = []
for _, row in df.iterrows():
    r_center = parse_ranges(row['center (km)'])
    r_strip1 = parse_ranges(row['strip_1 (km)'])
    r_strip2 = parse_ranges(row['strip_2 (km)'])
    u = union_ranges(r_center, r_strip1, r_strip2)
    if u:
        u_str = "; ".join([f"{s}~{e}" for s, e in u])
    else:
        u_str = ""
    unions.append(u_str)

df['center&strip1/2'] = unions

# intersection_center
intersections_center = []
for _, row in df.iterrows():
    r_center = parse_ranges(row['center (km)'])
    r_lst = parse_ranges(row['lst_difference_km'])
    inter = intersect_ranges(r_center, r_lst)
    inter_str = "; ".join([f"{s}~{e}" for s, e in inter]) if inter else ""
    intersections_center.append(inter_str)

df['intersection_center'] = intersections_center

# intersection_center&strip1/2
intersections_union = []
for _, row in df.iterrows():
    r_union = parse_ranges(row['center&strip1/2'])
    r_lst = parse_ranges(row['lst_difference_km'])
    inter = intersect_ranges(r_union, r_lst)
    inter_str = "; ".join([f"{s}~{e}" for s, e in inter]) if inter else ""
    intersections_union.append(inter_str)

df['intersection_center&strip1/2'] = intersections_union

# 计算区间范围
df['range_intersection_center'] = df['intersection_center'].apply(extract_range_sum)
df['range_intersection_center&strip1/2'] = df['intersection_center&strip1/2'].apply(extract_range_sum)
df['range_center&strip1/2'] = df['center&strip1/2'].apply(extract_range_sum)
df['range_center'] = df['center (km)'].apply(extract_range_sum)
# 保存
df.to_excel(output_path, index=False)
print(f"处理完成，结果已保存到：{output_path}")

