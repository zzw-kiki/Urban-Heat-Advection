import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import gaussian_kde

plt.rcParams["font.family"] = "Times New Roman"

# 文件路径
plot_dir = r"E:\UHA\images"
os.makedirs(plot_dir, exist_ok=True)
out_path = os.path.join(plot_dir, "Box_BD.png")

df = pd.read_csv(r"E:\UHA\city_BD.csv")  # city_name, BH_2010, BH_2020, size

# 添加 "all" 分类
df_all = df.copy()
df_all["size"] = "all"
df_plot = pd.concat([df, df_all], ignore_index=True)
# df_plot[["BD_2010", "BD_2020"]] = df[["BD_2010", "BD_2020"]] * 100
# 长表格
df_long = pd.melt(df_plot,
                  id_vars=["city_name", "size"],
                  value_vars=["BD_2010", "BD_2020"],
                  var_name="Year",
                  value_name="BD_Value")
df_long["BD_Value"] = df_long["BD_Value"]  * 100
# 设置 size 顺序和颜色
size_order = ["small", "middle", "large", "mega","all"]
colors = ['#264653', '#2A9D8E', '#E9C46B', '#E66F51','#8D99AE']

fig, ax = plt.subplots(figsize=(8, 6))
width = 0.2  # 每个箱子的宽度
offset = 0.2  # 两年份箱子左右偏移量

for i, size in enumerate(size_order):
    for j, year in enumerate(["BD_2010", "BD_2020"]):
        # 箱子横坐标
        x_pos = i + (-0.5 + j) * offset * 2  # 两年份微偏移

        # 对应数据
        y_vals = df_long[(df_long["size_3"] == size) & (df_long["Year"] == year)]["BD_Value"].dropna()
        if len(y_vals) == 0:
            continue

        # 绘制箱子
        bp = ax.boxplot(y_vals,
                        positions=[x_pos],
                        widths=width,
                        patch_artist=True,
                        boxprops=dict(facecolor=colors[i], alpha=1),
                        medianprops=dict(color='black'),
                        whiskerprops=dict(color='black'),
                        capprops=dict(color='black'),
                        flierprops=dict(marker='o', color='red', alpha=0.6),
                        showfliers=False)

        # 均值点
        mean_val = y_vals.mean()
        print(mean_val)
        ax.scatter(x_pos, mean_val, color='black', marker='s', s=20, zorder=10,label='Mean'if i==0 and j==0 else "")
        # ax.text(x_pos, mean_val + 0.6, f"{mean_val:.2f}", ha='center', va='bottom', fontsize=8)
# bh+0.6
        # KDE 曲线
        if len(y_vals) > 1:
            kde = gaussian_kde(y_vals)
            y_kde = np.linspace(y_vals.min(), y_vals.max(), 200)
            kde_vals = kde(y_kde)
            kde_vals = kde_vals / kde_vals.max() * 0.2  # 横向宽度
            x_start = x_pos + width/2
            x_vals = x_start + kde_vals
            ax.plot(x_vals, y_kde, color=colors[i], linewidth=2)
            ax.fill_betweenx(y_kde, x_start, x_vals, facecolor=colors[i], alpha=0.3)

# X 轴设置
ax.set_xticks(range(len(size_order)))
ax.set_xticklabels(size_order)
ax.set_xlabel("City Size", fontsize=16)
ax.set_ylabel("Building Density(%)", fontsize=16)
# ax.set_title("BH_2010 and BH_2020 by City Size")
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=12)
ax.legend(loc='upper left', fontsize=12)

ax.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(out_path, dpi=600, bbox_inches='tight')
plt.show()
