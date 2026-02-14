
import pandas as pd
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib.pyplot as plt
import os

import pandas as pd
import matplotlib.pyplot as plt
import os

import pandas as pd
import matplotlib.pyplot as plt
import os

import pandas as pd
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy.stats import gaussian_kde
import numpy as np
plt.rcParams["font.family"] = "Times New Roman"

# 文件路径
data_path = r"E:\UHA\climate_merged.csv"
out_dir = r"E:\UHA\images"
os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv(data_path, encoding="utf-8")

# 中文转英文
climate_map = {
    "暖温带半湿润地区": "Warm Temperate Semi-Humid Region",
    "中温带干旱地区": "Middle Temperate Arid Region",
    "北亚热带湿润地区": "Northern Subtropical Humid Region",
    "中温带半湿润地区": "Middle Temperate Semi-Humid Region",
    "中温带半干旱地区": "Middle Temperate Semi-Arid Region",
    "高原温带半干旱地区": "Plateau Temperate Semi-Arid Region",
    "边缘热带湿润地区": "Marginal Tropical Humid Region"
}
terrain_map = {"第一阶梯": "First Step", "第二阶梯": "Second Step", "第三阶梯": "Third Step"}
region_map = {"left": "West", "right": "East"}
size_map = {"small": "Small", "middle": "Middle", "large": "Large","Mega": "Mega"}

df["climate"] = df["climate"].map(climate_map)
df["terrain"] = df["terrain"].map(terrain_map)
df["Huhuanyong"] = df["Huhuanyong"].map(region_map)
df["size"] = df["size"].map(size_map)
value_cols = ["UHAE_temp_3_Modified",
              "UHAE_range"]

colors = ["#4C72B0", "#DD8452"]
bar_height = 0.6

# 按三个分类列分别计算平均值，并依次合并
avg_list = []
category_labels = []

for col in ["climate", "terrain", "Huhuanyong","size"]:
    avg = df.groupby(col)[value_cols].mean()
    # 打印结果
    print(f"\n=== {col} ===")
    for idx, row in avg.iterrows():
        print(f"{idx}: UHA intensity = {row[value_cols[0]]:.2f} ℃, "
              f"UHA Range = {row[value_cols[1]]:.2f} km")
    avg_list.append(avg)
    category_labels.extend(avg.index.tolist())

# 合并所有平均值
avg_thermal = pd.concat([avg[value_cols[0]] for avg in avg_list])
avg_range = pd.concat([avg[value_cols[1]] for avg in avg_list])
category_labels = category_labels[::-1]
avg_thermal = avg_thermal[::-1]
avg_range = avg_range[::-1]

cols = ["climate", "terrain", "Huhuanyong", "size"]

ordered_labels = []
grouped_data = []
grouped_data2 = []

# 构建标签和箱型数据
for col, avg in zip(cols, avg_list):
    if col == "size":
        # 指定你想要的顺序
        avg = avg.reindex(["Mega","Large", "Middle", "Small"])
    names = list(avg.index)
    for name in names:
        ordered_labels.append(f"{name}")  # 标签带分类名
        grouped_data.append(df.loc[df[col] == name, value_cols[0]].dropna())
        grouped_data2.append(df.loc[df[col] == name, value_cols[1]].dropna())

# 如果需要反转顺序
ordered_labels = ordered_labels[::-1]
grouped_data = grouped_data[::-1]
grouped_data2 = grouped_data2[::-1]

# 绘图
fig, axes = plt.subplots(1, 2, figsize=(14, max(7, 0.3*len(ordered_labels))), sharey=False)
width = 0.5
# 注意 boxplot 默认位置从 1 开始
y_pos = list(range(1, len(ordered_labels)+1))

# 左图
axes[0].boxplot(grouped_data, vert=False, patch_artist=True,
                boxprops=dict(facecolor=colors[0], color='none'),
                medianprops=dict(color='black'), showfliers=False,
                )
axes[0].set_yticks(y_pos)
axes[0].set_yticklabels(ordered_labels, fontsize=10)
# axes[0].invert_yaxis()
axes[0].set_xlabel("Thermal intensity of the ΔUHA effect (℃)", fontsize=12)
axes[0].spines["right"].set_visible(False)
axes[0].spines["top"].set_visible(False)

# KDE叠加
for i, x_vals in enumerate(grouped_data):
    x_vals = np.array(x_vals)
    if len(x_vals) > 1:
        kde = gaussian_kde(x_vals)
        x_kde = np.linspace(x_vals.min(), x_vals.max(), 200)
        y_kde = kde(x_kde)
        y_kde = y_kde / y_kde.max() * width  # 缩放到合适的高度
        y_center = y_pos[i]  # KDE纵向位置
        axes[0].plot(x_kde, y_center + y_kde, color='gray', linewidth=0)
        axes[0].fill_between(x_kde, y_center, y_center + y_kde, color='gray', alpha=0.15)

# 右图
axes[1].boxplot(grouped_data2, vert=False, patch_artist=True,
                boxprops=dict(facecolor=colors[1], color='none'),
                medianprops=dict(color='black'), showfliers=False)
axes[1].set_yticks(y_pos)
axes[1].set_yticklabels([])  # 不显示右图标签
# axes[1].invert_yaxis()
axes[1].set_xlabel("Spatial extent of the ΔUHA effect (km)", fontsize=12)
axes[1].spines["right"].set_visible(False)
axes[1].spines["top"].set_visible(False)
# KDE叠加
for i, x_vals in enumerate(grouped_data2):
    x_vals = np.array(x_vals)
    if len(x_vals) > 1:
        kde = gaussian_kde(x_vals)
        x_kde = np.linspace(x_vals.min(), x_vals.max(), 200)
        y_kde = kde(x_kde)
        y_kde = y_kde / y_kde.max() * width  # 缩放到合适的高度
        y_center = y_pos[i]  # KDE纵向位置

        axes[1].plot(x_kde, y_center + y_kde, color='gray', linewidth=0)
        axes[1].fill_between(x_kde, y_center, y_center + y_kde, color='gray', alpha=0.15)# 不加底色前 alpha=0.15
# 左图添加均值点
for i, x_vals in enumerate(grouped_data):
    x_vals = np.array(x_vals)
    if len(x_vals) > 0:
        mean_val = x_vals.mean()
        axes[0].scatter(mean_val, y_pos[i], color='black', marker='o', s=10, zorder=5)  # 红色菱形点

# 右图添加均值点
for i, x_vals in enumerate(grouped_data2):
    x_vals = np.array(x_vals)
    if len(x_vals) > 0:
        mean_val = x_vals.mean()
        axes[1].scatter(mean_val, y_pos[i], color='black', marker='o', s=10, zorder=5)

background_colors = {
    "climate": "#ffbd7c",      # 浅灰
    "terrain": "#82b1d3",      # 浅绿
    "Huhuanyong": "#fa7f6f",   # 浅蓝
    "size": "#8dcec9"          # 浅橙
}
# 每个分类对应的行数
group_lengths = [len(avg) for avg in avg_list]

# 当前标签是倒序的，所以也要倒序遍历
y_bottom = 0
for (col, length) in zip(cols[::-1], group_lengths[::-1]):
    y_top = y_bottom + length
    for ax in axes:  # 两张图都加背景
        ax.axhspan(y_bottom + 0.5, y_top + 0.5,
                   color=background_colors[col], alpha=0.5, zorder=0)
    y_bottom = y_top

plt.tight_layout()

plt.savefig(os.path.join(out_dir, "Climate_statistic.png"), dpi=600, bbox_inches="tight")
plt.show()

print("统计完成，图已保存到：", out_dir)





