import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
import os
# UHA
# ---------------- 文件路径 ----------------
uha_bh_path = r"E:\UHAE\BH&BD\city_BH.csv"
regression_paths = {
    "center": r"E:\UHAE\regression\3km_1km_buffer_center\regression_CPR.csv",
    "strip1": r"E:\UHAE\regression\3km_1km_buffer_strip1\regression_CPR.csv",
    "strip2": r"E:\UHAE\regression\3km_1km_buffer_strip2\regression_CPR.csv",
}
final_lst_path = r"E:\UHAE\LST_distance\final_lst_distance.csv"
output_path = r"E:\UHAE\UHA_result\UHA.csv"

# ---------------- 读取 CSV ----------------
uha_df = pd.read_csv(uha_bh_path)
lst_df = pd.read_csv(final_lst_path)

# ---------------- 初始化新列 ----------------
for source in ['center', 'strip1', 'strip2']:
    for year in ['2010', '2020']:
        uha_df[f'range_{year}_{source}'] = ""  # 用区间字符串表示
        uha_df[f'temp_{year}_{source}'] = np.nan

uha_df['BH_2010'] = np.nan
uha_df['BH_2020'] = np.nan

def calculate_range_only(dist_new, resid_new):
    positive_mask = resid_new > 0
    intervals = []

    start_idx = None
    for i, val in enumerate(positive_mask):
        if val and start_idx is None:
            start_idx = i
        elif not val and start_idx is not None:
            intervals.append((dist_new[start_idx], dist_new[i - 1]))
            start_idx = None

    if start_idx is not None:
        intervals.append((dist_new[start_idx], dist_new[-1]))

    interval_str = "; ".join([f"{round(s,2)}~{round(e,2)}" for s, e in intervals])
    return interval_str


# ---------------- 遍历每个城市 ----------------
for idx, row in uha_df.iterrows():
    city = row['city_name']

    for source, reg_path in regression_paths.items():
        city_data = pd.read_csv(reg_path)
        city_data = city_data[city_data['city_name'] == city]

        if city_data.empty:
            continue

        # BH 值只取 center
        if source == 'center':
            bh0 = city_data[city_data['distance'] == 0]
            if not bh0.empty:
                uha_df.at[idx, 'BH_2010'] = bh0['BH_2010'].values[0]
                uha_df.at[idx, 'BH_2020'] = bh0['BH_2020'].values[0]

        for year in ['2010', '2020']:
            dist = city_data['distance'][city_data['distance'] >= 0].values
            resid = city_data[f'residual_{year}'][city_data['distance'] >= 0].values

            if len(dist) < 2:
                continue

            sorted_idx = np.argsort(dist)
            dist_sorted = dist[sorted_idx]
            resid_sorted = resid[sorted_idx]

            spline = UnivariateSpline(dist_sorted, resid_sorted, s=0)
            dist_new = np.linspace(dist_sorted.min(), dist_sorted.max(), 300)
            resid_new = spline(dist_new)

            # 只计算区间字符串，不计算 temp
            range_str = calculate_range_only(dist_new, resid_new)
            uha_df.at[idx, f'range_{year}_{source}'] = range_str


# ---------------- 合并 LST 数据 ----------------
merged_df = pd.merge(uha_df, lst_df[['city','union_lst_difference_km']], left_on='city_name', right_on='city', how='left')

# ---------------- 区间处理函数 ----------------
def parse_intervals(interval_str):
    """ '1.2~3.5; 4.0~5.8' -> [(1.2,3.5),(4.0,5.8)] """
    if pd.isna(interval_str) or str(interval_str).strip() == "":
        return []
    intervals = []
    for seg in str(interval_str).split(';'):
        s, e = seg.strip().split('~')
        intervals.append((float(s), float(e)))
    return intervals

def union_intervals(interval_list):
    if not interval_list:
        return []
    interval_list = sorted(interval_list, key=lambda x: x[0])
    merged = [interval_list[0]]
    for s, e in interval_list[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e:
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))
    return merged

def intersect_intervals(interval_list1, interval_list2):
    """计算两个区间列表的交集"""
    result = []
    for s1, e1 in interval_list1:
        for s2, e2 in interval_list2:
            start = max(s1, s2)
            end = min(e1, e2)
            if start < end:
                result.append((start, end))
    return union_intervals(result)

def intervals_to_str(interval_list):
    return "; ".join([f"{round(s,2)}~{round(e,2)}" for s,e in interval_list])

def intervals_length(interval_list):
    return sum(e-s for s,e in interval_list)

# ---------------- 计算 UHA 区间和长度 ----------------
for year in ['2010', '2020']:
    uha_range_col = f'uha_range_interval_{year}'
    uha_length_col = f'uha_range_{year}'
    uha_temp_col = f'temp_{year}'  # 最终 temp 列


    def compute_uha_intervals_and_temp(row):
        temps_per_source = []
        final_intervals = []
        temps_dict = {}  # 保存源 temp

        lst_intervals = parse_intervals(row.get('union_lst_difference_km', ""))

        for source in ['center', 'strip1', 'strip2']:
            col_name = f'range_{year}_{source}'
            source_intervals = parse_intervals(row.get(col_name, ""))

            intersected = intersect_intervals(source_intervals, lst_intervals)

            temp_val = 0
            if intersected:
                reg_path = regression_paths[source]
                city_data = pd.read_csv(reg_path)
                city_data = city_data[city_data['city_name'] == row['city_name']]
                if not city_data.empty:
                    dist = city_data['distance'][city_data['distance'] >= 0].values
                    resid = city_data[f'residual_{year}'][city_data['distance'] >= 0].values
                    if len(dist) >= 2:
                        sorted_idx = np.argsort(dist)
                        dist_sorted = dist[sorted_idx]
                        resid_sorted = resid[sorted_idx]
                        spline = UnivariateSpline(dist_sorted, resid_sorted, s=0)
                        dist_new = np.linspace(dist_sorted.min(), dist_sorted.max(), 300)
                        resid_new = spline(dist_new)

                        temp_vals = []
                        for s, e in intersected:
                            mask = (dist_new >= s) & (dist_new <= e)
                            temp_vals.extend(resid_new[mask])
                        temp_val = np.mean(temp_vals) if temp_vals else 0

            # 保存到字典
            temps_dict[f'temp_{year}_{source}'] = temp_val
            temps_per_source.append(temp_val)
            final_intervals.extend(intersected)

        final_temp = np.mean(temps_per_source) if temps_per_source else 0
        merged_final = union_intervals(final_intervals)

        # 返回最终 temp 和每个源 temp
        return intervals_to_str(merged_final), intervals_length(merged_final), final_temp, \
               temps_dict[f'temp_{year}_center'], temps_dict[f'temp_{year}_strip1'], temps_dict[f'temp_{year}_strip2']


    merged_df[[uha_range_col, uha_length_col, uha_temp_col,
               f'temp_{year}_center', f'temp_{year}_strip1', f'temp_{year}_strip2']] = \
        merged_df.apply(lambda r: pd.Series(compute_uha_intervals_and_temp(r)), axis=1)

# 删除冗余 city 列
if 'city' in merged_df.columns:
    merged_df.drop(columns=['city'], inplace=True)

# ---------------- 保存最终结果 ----------------
os.makedirs(os.path.dirname(output_path), exist_ok=True)
merged_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print("✅ 已生成最终结果:", output_path)

# UHAE
'''
import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
import os

# ---------------- 文件路径 ----------------
uha_bh_path = r"E:\UHAE\BH&BD\city_BH.csv"
regression_paths = {
    "center": r"E:\UHAE\regression\3km_1km_buffer_center\regression_CPR.csv",
    "strip1": r"E:\UHAE\regression\3km_1km_buffer_strip1\regression_CPR.csv",
    "strip2": r"E:\UHAE\regression\3km_1km_buffer_strip2\regression_CPR.csv",
}
final_lst_path = r"E:\UHAE\LST_distance\final_lst_distance.csv"
output_path = r"E:\UHAE\UHA_result\UHAE.csv"

# ---------------- 读取 CSV ----------------
uha_df = pd.read_csv(uha_bh_path)
lst_df = pd.read_csv(final_lst_path)

# ---------------- 初始化新列 ----------------
for source in ['center','strip1','strip2']:
    uha_df[f'range_{source}'] = ""
    uha_df[f'temp_{source}'] = np.nan

uha_df['UHAE_range'] = ""
uha_df['UHAE_temp'] = np.nan

# ---------------- 计算正 residual 差区间 ----------------
def calculate_range_only(dist_new, resid_diff):
    positive_mask = resid_diff > 0
    intervals = []
    start_idx = None
    for i, val in enumerate(positive_mask):
        if val and start_idx is None:
            start_idx = i
        elif not val and start_idx is not None:
            intervals.append((dist_new[start_idx], dist_new[i-1]))
            start_idx = None
    if start_idx is not None:
        intervals.append((dist_new[start_idx], dist_new[-1]))
    return "; ".join([f"{round(s,2)}~{round(e,2)}" for s,e in intervals])

# ---------------- 区间处理函数 ----------------
def parse_intervals(interval_str):
    if pd.isna(interval_str) or str(interval_str).strip() == "":
        return []
    intervals = []
    for seg in str(interval_str).split(';'):
        s, e = seg.strip().split('~')
        intervals.append((float(s), float(e)))
    return intervals

def union_intervals(interval_list):
    if not interval_list: return []
    interval_list = sorted(interval_list, key=lambda x:x[0])
    merged = [interval_list[0]]
    for s,e in interval_list[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e:
            merged[-1] = (last_s, max(last_e,e))
        else:
            merged.append((s,e))
    return merged

def intersect_intervals(interval_list1, interval_list2):
    result = []
    for s1,e1 in interval_list1:
        for s2,e2 in interval_list2:
            start = max(s1,s2)
            end = min(e1,e2)
            if start < end:
                result.append((start,end))
    return union_intervals(result)

def intervals_to_str(interval_list):
    return "; ".join([f"{round(s,2)}~{round(e,2)}" for s,e in interval_list])

def intervals_length(interval_list):
    return sum(e-s for s,e in interval_list)

# ---------------- 遍历每个城市 ----------------
for idx, row in uha_df.iterrows():
    city = row['city_name']

    for source, reg_path in regression_paths.items():
        city_data = pd.read_csv(reg_path)
        city_data = city_data[city_data['city_name']==city]
        if city_data.empty: 
            continue

        # 计算 residual 差值：2020-2010
        dist = city_data['distance'][city_data['distance']>=0].values
        resid_diff = city_data['residual_2020'][city_data['distance']>=0].values - \
                     city_data['residual_2010'][city_data['distance']>=0].values
        if len(dist) < 2:
            continue

        sorted_idx = np.argsort(dist)
        dist_sorted = dist[sorted_idx]
        resid_diff_sorted = resid_diff[sorted_idx]

        spline = UnivariateSpline(dist_sorted, resid_diff_sorted, s=0)
        dist_new = np.linspace(dist_sorted.min(), dist_sorted.max(), 300)
        resid_new = spline(dist_new)

        # 计算正 residual 差区间
        range_str = calculate_range_only(dist_new, resid_new)
        uha_df.at[idx,f'range_{source}'] = range_str

# ---------------- 合并 LST 数据 ----------------
merged_df = pd.merge(uha_df, lst_df[['city','union_lst_difference_km']], 
                     left_on='city_name', right_on='city', how='left')
merged_df.drop(columns=['city'], inplace=True)

# ---------------- 计算 UHAE 区间和 temp ----------------
def compute_uhae_intervals_and_temp(row):
    temps_per_source = []
    final_intervals = []

    lst_intervals = parse_intervals(row.get('union_lst_difference_km',""))

    for source in ['center','strip1','strip2']:
        col_name = f'range_{source}'
        source_intervals = parse_intervals(row.get(col_name,""))

        # 与 LST 交集
        intersected = intersect_intervals(source_intervals, lst_intervals)

        temp_val = 0
        if intersected:
            # 从 regression 文件重新取 residual 差
            reg_path = regression_paths[source]
            city_data = pd.read_csv(reg_path)
            city_data = city_data[city_data['city_name']==row['city_name']]
            if not city_data.empty:
                dist = city_data['distance'][city_data['distance']>=0].values
                resid_diff = city_data['residual_2020'][city_data['distance']>=0].values - \
                             city_data['residual_2010'][city_data['distance']>=0].values
                if len(dist) >= 2:
                    sorted_idx = np.argsort(dist)
                    dist_sorted = dist[sorted_idx]
                    resid_diff_sorted = resid_diff[sorted_idx]
                    spline = UnivariateSpline(dist_sorted, resid_diff_sorted, s=0)
                    dist_new = np.linspace(dist_sorted.min(), dist_sorted.max(), 300)
                    resid_new = spline(dist_new)

                    temp_vals = []
                    for s,e in intersected:
                        mask = (dist_new>=s) & (dist_new<=e)
                        temp_vals.extend(resid_new[mask])
                    temp_val = np.mean(temp_vals) if temp_vals else 0

        # 保存源 temp
        row[f'temp_{source}'] = temp_val
        temps_per_source.append(temp_val)
        final_intervals.extend(intersected)

    # 最终 UHAE temp
    final_temp = np.mean(temps_per_source) if temps_per_source else 0
    merged_final = union_intervals(final_intervals)

    return intervals_to_str(merged_final), final_temp

merged_df[['UHAE_range','UHAE_temp']] = merged_df.apply(
    lambda r: pd.Series(compute_uhae_intervals_and_temp(r)), axis=1
)

# ---------------- 保存结果 ----------------
os.makedirs(os.path.dirname(output_path), exist_ok=True)
merged_df.to_csv(output_path,index=False,encoding='utf-8-sig')
print("✅ 已生成 UHAE 结果:", output_path)
'''
