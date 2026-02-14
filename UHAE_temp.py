import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress,t
import os

# 全局字体
plt.rcParams["font.family"] = "Times New Roman"

# 文件路径
csv_path = r"E:\UHAE\UHA_result\final_UHAE_192.csv"
output_dir = r" "
os.makedirs(output_dir, exist_ok=True)

# 读取数据
df = pd.read_csv(csv_path)
print(df.columns.tolist())

# 根据 size 设置散点大小
size_map = {'small': 30, 'middle': 60, 'large': 100}
df['marker_size'] = df['size_3'].map(size_map)
'''
# 提取字符串区间中的最大值
def extract_max(s):
    if pd.isna(s):
        return np.nan
    parts = s.split(';')
    try:
        max_val = max(float(p.split('~')[1]) for p in parts)
    except:
        max_val = np.nan
    return max_val
'''
'''
# 提取字符串区间中的第一个区间的最大值
def extract_max(s):
    if pd.isna(s):
        return np.nan
    parts = s.split(';')
    try:
        # 只取第一个区间，并提取右边界
        max_val = float(parts[0].split('~')[1])
    except:
        max_val = np.nan
    return max_val
'''
# 提取字符串区间的范围总和（右端 - 左端）
def extract_range_sum(s):
    if pd.isna(s):
        return np.nan
    parts = s.split(';')
    try:
        total_range = 0.0
        for p in parts:
            bounds = p.split('~')
            if len(bounds) == 2:
                left = float(bounds[0])
                right = float(bounds[1])
                total_range += (right - left)
        return total_range if total_range > 0 else np.nan
    except:
        return np.nan

y_cols = ['UHAE_temp']

x_all = df['center_bh_rise(m)'].values # +df['strip1_bh_rise(m)'].values+df['strip2_bh_rise(m)'].values


for col in y_cols:
    # y_all = df[col+'_range'].values
    y_all = df[col].values
    # 剔除 NaN
    valid_mask_all = (~np.isnan(y_all))
    x_valid_all = x_all[valid_mask_all]
    y_valid_all = y_all[valid_mask_all]
    size_valid_all = df['marker_size'][valid_mask_all]
    size_category_all = df['size_3'][valid_mask_all]
    # 全局 X 范围
    global_x_min = np.nanmin(x_valid_all)
    global_x_max = np.nanmax(x_valid_all)
    global_line_x = np.linspace(global_x_min, global_x_max, 200)

    plt.figure(figsize=(8, 6))
    colors = {'small': '#31a3a2', 'middle': '#0e6090', 'large': '#C62B13', 'all': 'black'}
    # 绘制三类散点
    for sz_label, sz_value in size_map.items():
        mask = size_category_all == sz_label
        plt.scatter(x_valid_all[mask], y_valid_all[mask],
                    s=size_valid_all[mask],
                    c=colors[sz_label],
                    linewidths= 1,
                    # edgecolors='none',
                    alpha=0.8,
                    label=sz_label.capitalize())

    # 存储每条拟合线的文本信息
    text_lines = []

    # 分三类分别拟合


    for category in ['small', 'middle', 'large', 'all']:
        if category == 'all':
            x_fit = x_valid_all
            y_fit = y_valid_all
        else:
            mask_cat = size_category_all == category
            x_fit = x_valid_all[mask_cat]
            y_fit = y_valid_all[mask_cat]

        if len(x_fit) < 2:
            continue

        slope, intercept, r_value, p_value, std_err = linregress(x_fit, y_fit)
        r_squared = r_value**2
        # 计算置信区间（95%）
        dfree = len(x_fit) - 2
        t_crit = t.ppf(0.975, dfree)  # 双侧95%
        slope_ci = (slope - t_crit * std_err, slope + t_crit * std_err)

        # 计算截距标准误（按线性回归公式估计）
        y_pred = slope * x_fit + intercept
        residual_std = np.sqrt(np.sum((y_fit - y_pred) ** 2) / dfree)
        x_mean = np.mean(x_fit)
        se_intercept = residual_std * np.sqrt(np.sum(x_fit ** 2) / (len(x_fit) * np.sum((x_fit - x_mean) ** 2)))
        intercept_ci = (intercept - t_crit * se_intercept, intercept + t_crit * se_intercept)

        # 打印置信区间
        print(f"\n===== {category.upper()} 回归结果 =====")
        print(f"Slope = {slope:.4f}  (95% CI: {slope_ci[0]:.4f}, {slope_ci[1]:.4f})")
        print(f"Intercept = {intercept:.4f}  (95% CI: {intercept_ci[0]:.4f}, {intercept_ci[1]:.4f})")
        print(f"R² = {r_squared:.4f}, R = {r_value:.4f}, n = {len(x_fit)}")

        line_x = np.linspace(np.nanmin(x_fit), np.nanmax(x_fit), 100)
        line_y = slope * global_line_x + intercept

        plt.plot(global_line_x, line_y, color=colors[category], linewidth=2,linestyle="--" if category == "all" else "-",
                 label=f"{category.capitalize()} fit: y={slope:.2f}x+{intercept:.2f}")

        info = f"{category.capitalize()} R²={r_squared:.3f}, p={p_value:.3g}" #, p={p_value:.3g}"
        text_lines.append(info)

        # 👉 在控制台打印
        print(f"{col} - {info}")

    # 标注文本（放在右上角）
    plt.text(0.65, 0.2, "\n".join(text_lines),
             transform=plt.gca().transAxes,
             fontsize=14, verticalalignment='top',
             horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))

    #plt.xlabel('Center Building Height Rise (m)')
    #plt.ylabel(f"UHA Intensity (℃)")
    # plt.ylabel(f"WBTI")
    # plt.title(f'Linear Regression: Thermal lag temp rise vs Center BH Rise')
    plt.legend(loc='upper left', bbox_to_anchor=(0, 1),fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'BH_temp_fit.png'), dpi=600)
    # plt.close()
    plt.show()

print(f"回归图已保存至：{output_dir}")
