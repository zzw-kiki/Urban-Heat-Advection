# 绘制上下风口温差曲线，并计算温差大于0即下风口温度大于上风口温度的距离段
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import os
import matplotlib
matplotlib.use('Agg')

from matplotlib import rcParams
plt.rcParams["font.family"] = "Times New Roman"
# 文件和输出目录映射
file_map = {
    "center": (
        r"E:\UHAE\UHA_result\UHA_3km_1km_center.csv",
        r"E:\UHAE\LST_distance\center"
    ),
    "strip1": (
        r"E:\UHAE\UHA_result\UHA_3km_1km_strip1.csv",
        r"E:\UHAE\LST_distance\strip1"
    ),
    "strip2": (
        r"E:\UHAE\UHA_result\UHA_3km_1km_strip2.csv",
        r"E:\UHAE\LST_distance\strip2"
    ),
}

output_csv_map = {
    "center": r"E:\UHAE\LST_distance\center\lst_difference.csv",
    "strip1": r"E:\UHAE\LST_distance\strip1\lst_difference.csv",
    "strip2": r"E:\UHAE\LST_distance\strip2\lst_difference.csv",
}

uha_data = {}

# ---------- 即存在下风口温度大于上风口温度的城市256座 ----------
center_df = pd.read_csv(file_map['center'][0])
valid_cities = center_df.loc[center_df['ΔT2020_MAX_value'] > 0, 'city'].tolist()

# x 轴（距离）
x_vals = np.arange(1, 31)

for name, (input_file, output_dir) in file_map.items():
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_file)
    df = df[df['city'].isin(valid_cities)]
    # 遍历每一个城市
    for _, row in df.iterrows():
        city = row['city']

        # 提取列名
        avg_cols = [f'ΔT2020_{i}km' for i in x_vals]
        avg_values = row[avg_cols].values.astype(float)

        fig, ax1 = plt.subplots(figsize=(8, 6))

        # 画 LST_Distance 曲线（左轴）
        valid_avg = ~np.isnan(avg_values)
        x_avg = x_vals[valid_avg]
        y_avg = avg_values[valid_avg]

        if len(x_avg) > 1:
            x_new = np.linspace(x_avg.min(), x_avg.max(), 300)
            y_smooth = make_interp_spline(x_avg, y_avg)(x_new)
            ax1.plot(x_new, y_smooth, color='#4EBEFF', label='Cumulative LST Difference')

            # 统计平滑后的上下风口温差> 0的所有连续x范围
            mask = y_smooth > 0
            x_above_05 = x_new[mask]
            # 识别连续区间（根据间距差异是否超过一定阈值）
            ranges = []
            if len(x_above_05) > 0:
                group = [x_above_05[0]]
                for i in range(1, len(x_above_05)):
                    if x_above_05[i] - x_above_05[i - 1] <= (x_new[1] - x_new[0]) * 2:  # 连续间距
                        group.append(x_above_05[i])
                    else:
                        ranges.append((round(group[0], 2), round(group[-1], 2)))
                        group = [x_above_05[i]]
                ranges.append((round(group[0], 2), round(group[-1], 2)))  # 添加最后一组

            # 构造 range 字符串，如 1.2~5.4; 10.0~25.1
            range_str = "; ".join([f"{start}~{end}" for start, end in ranges]) if ranges else ""

            # 保存结果到一个列表中备用
            if 'uha_data' not in locals():
                uha_data = {}
            if name not in uha_data:
                uha_data[name] = []
            uha_data[name].append({
                'city': city,
                f'{name}_lst_difference_km': range_str
            })
        else:
            ax1.plot(x_avg, y_avg, color='#4EBEFF', label='Cumulative Average LST')
        ax1.plot(x_avg, y_avg, 'o', color='#4EBEFF', markersize=3, zorder=4)
        # 添加 y=0 处的红色虚线
        ax1.axhline(y=0, color='red', linestyle='--', linewidth=1)
        ax1.set_xlabel("Downwind Distance (km)")
        ax1.set_ylabel("Cumulative LST Difference (℃)", color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.set_xlim(1, 30)
        # 标题与图例
        plt.title(f" LST Difference - {name}")
        lines1, labels1 = ax1.get_legend_handles_labels()
        # lines2, labels2 = ax2.get_legend_handles_labels()
        fig.legend(lines1 , labels1 , loc='upper left', bbox_to_anchor=(0, 1),
                   bbox_transform=ax1.transAxes, fontsize=9)
        fig.tight_layout()

        # 保存图像
        out_path = os.path.join(output_dir, f"LST Difference of {city} - {name}.png")
        plt.savefig(out_path, dpi=200)
        plt.close()
    print(f"所有城市图像已保存至: {output_dir}")
for name, records in uha_data.items():
    df_output = pd.DataFrame(records)
    df_output.to_csv(output_csv_map[name], index=False, encoding='utf-8-sig')
    print(f"{name} 上下风口温差范围统计已保存到：{output_csv_map[name]}")

# ================= 合并 center / strip1 / strip2  =================
import pandas as pd
import numpy as np
import os
from functools import reduce

def parse_range_str(range_str):
    """ '1.2~5.4; 10.0~25.1' -> [(1.2,5.4),(10.0,25.1)] """
    if pd.isna(range_str) or str(range_str).strip() == "":
        return []
    intervals = []
    for seg in str(range_str).split(';'):
        start, end = seg.strip().split('~')
        intervals.append((float(start), float(end)))
    return intervals


def union_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0]]

    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:   # 重叠或相接
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def intervals_to_str(intervals):
    """ [(1.2,25.1),(26.0,28.0)] -> '1.2~25.1; 26.0~28.0' """
    if not intervals:
        return ""
    return "; ".join([f"{s:.1f}~{e:.1f}" for s, e in intervals])


def compute_math_union(row):
    all_intervals = []
    for col in [
        'center_lst_difference_km',
        'strip1_lst_difference_km',
        'strip2_lst_difference_km'
    ]:
        if col in row:
            all_intervals.extend(parse_range_str(row[col]))

    merged = union_intervals(all_intervals)
    return intervals_to_str(merged)

dfs = []
for name, records in uha_data.items():
    dfs.append(pd.DataFrame(records))

final_df = reduce(
    lambda left, right: pd.merge(left, right, on='city', how='outer'),
    dfs
)

final_df['union_lst_difference_km'] = final_df.apply(
    compute_math_union, axis=1
)

final_output_dir = r"E:\UHAE\LST_distance"
os.makedirs(final_output_dir, exist_ok=True)

final_csv_path = os.path.join(final_output_dir, "final_lst_distance.csv")
final_df.to_csv(final_csv_path, index=False, encoding='utf-8-sig')

print("✅ 已生成最终合并结果：", final_csv_path)
