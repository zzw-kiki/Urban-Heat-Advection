import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
import os
from scipy.stats import linregress, t
plt.rcParams["font.family"] = "Times New Roman"
# ================= 读取数据 =================
file = r"E:\UHA\UHA_BH_All_with_wind_difference.csv"
# file = r"E:\WBTI\筛选结果\论文\UHA\UHA_BH&speed_updated_2010_range_modified.csv"
df = pd.read_csv(file)
df = df[df['风向差(绝对值)'] <= 45]
# ================= 设置颜色与标签 =================
colors = {
    'Mega': '#FFC43A',
    'large': '#f3474b',
    'middle': '#31a3a2',
    'small': '#0e6090',
}
labels = {
    'Mega': 'Mega',
    'large': 'Large',
    'middle': 'Medium',
    'small': 'Small',
}
# 平均点的符号（黑色）
mean_markers = {
    'large': 'x',   # 叉叉
    'middle': '^',  # 三角形
    'small': 'o',   # 圆点
    'Mega': 's',   # 正方形
}

# ================= 定义绘图函数 =================
def plot_regression_overall(df,x_col, y_col, year, x_label, y_label, save_name):
    plt.figure(figsize=(6, 5))

    # --- 绘制不同 size 的散点 ---
    # --- 按顺序绘制散点 small → medium → large ---
    for size in ['small', 'middle', 'large', 'Mega']:
        group = df[df['size_1'] == size]
        if not group.empty:
            plt.scatter(group[x_col], group[y_col],
                        color=colors.get(size, 'gray'),
                        alpha=0.8, s=40,
                        label=labels.get(size, size.capitalize()))
    for size in ['small', 'middle', 'large', 'Mega']:
        group = df[df['size_1'] == size]
        if not group.empty:
            mean_x = group[x_col].mean()
            mean_y = group[y_col].mean()
            plt.scatter(mean_x, mean_y,
                        color='black',
                        marker=mean_markers[size],
                        s=50, linewidths=2,
                        label=f"{labels[size]} Mean")
    # --- 计算总体回归 ---
    x_all = df[x_col].values
    y_all = df[y_col].values
    mask = np.isfinite(x_all) & np.isfinite(y_all)
    x_all, y_all = x_all[mask], y_all[mask]

    if len(x_all) > 1:
        slope, intercept, r_value, p_value, std_err = linregress(x_all, y_all)
        x_fit = np.linspace(x_all.min(), x_all.max(), 200)
        y_fit = slope * x_fit + intercept
        plt.plot(x_fit, y_fit, color='black', linestyle='--', linewidth=1.2,
                 label=f'Overall fit ')
        # --- 显示回归方程 ---
        eq_text = f"y = {slope:.4f}x + {intercept:.2f}\nR² = {r_value ** 2:.2f}\np = {p_value:.3e}"
        plt.text(0.76, 0.2, eq_text, transform=plt.gca().transAxes,
                 fontsize=10, va='top', ha='left', color='black',
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.6))
        # === 计算回归系数置信区间（95%） ===
        n = len(x_all)
        alpha = 0.05
        t_val = t.ppf(1 - alpha / 2, df=n - 2)
        slope_ci = (slope - t_val * std_err, slope + t_val * std_err)
        intercept_std_err = std_err * np.sqrt(np.sum(x_all ** 2) / (n * np.var(x_all, ddof=1)))
        intercept_ci = (intercept - t_val * intercept_std_err,
                        intercept + t_val * intercept_std_err)

        print(f"\n【{year} 回归结果】")
        print(f"y = {slope:.4f}x + {intercept:.4f}")
        print(f"R² = {r_value ** 2:.4f}")
        print(f"R = {r_value:.4f}")
        print(f"斜率95%置信区间: {slope_ci[0]:.4f} ~ {slope_ci[1]:.4f}")
        print(f"截距95%置信区间: {intercept_ci[0]:.4f} ~ {intercept_ci[1]:.4f}")

    # --- 样式 ---
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.grid(alpha=0.3, linestyle='--')
    plt.legend(frameon=False, fontsize=10, loc='upper left')
    plt.tight_layout()

    # --- 保存 ---
    save_dir = r"E:\WBTI\筛选结果\论文\论文出图4"
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.show()
    print(f"✅ 图已保存：{save_path}")


# ================= 生成四张图 =================

plot_regression_overall(df,'BH_2010', 'temp_2010', 2010,
                        'Building Height 2010 (m)', 'UHA_2010 (°C)',
                        'UHA_2010_vs_BH2010.png')

plot_regression_overall(df,'BH_2010', 'range_2010', 2010,
                        'Building Height 2010 (m)', 'Spatial Extent of UHA phenomenon 2010 (km)',
                        'range2010_vs_BH2010.png')

plot_regression_overall(df,'BH_2020', 'temp_2020', 2020,
                        'Building Height 2020 (m)', 'UHA_2020 (°C)',
                        'UHA_2020_vs_BH2020.png')

plot_regression_overall(df,'BH_2020', 'range_2020', 2020,
                        'Building Height 2020 (m)', 'Spatial Extent of UHA phenomenon 2020 (km)',
                        'range2020_vs_BH2020.png')
