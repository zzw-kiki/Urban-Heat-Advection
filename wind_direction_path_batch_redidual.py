import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.interpolate import make_interp_spline
import numpy as np
import os
from matplotlib import rcParams

failed_city = []
plt.rcParams["font.family"] = "Times New Roman"
# 让 matplotlib 中文显示用宋体，英文用 Times New Roman
rcParams['font.sans-serif'] = ['SimSun']  # 中文宋体
plt.rcParams["font.family"] = "Times New Roman"
rcParams['axes.unicode_minus'] = False  # 负号正常显示
# 读取数据
# df = pd.read_csv(r"E:\WBTI\wind_path_statistic\3km_3km_buffer\data.csv")
df = pd.read_csv(r"E:\UHA\regression\3km_1km_buffer_strip1\regression_CPR.csv")
# 新建输出文件夹（如果不存在）
# output_dir = r"E:\WBTI\figure_result\figure_result_3km_center"
output_dir = r"E:\UHA\regression\3km_1km_buffer_strip1\predict_result_CPR"
os.makedirs(output_dir, exist_ok=True)
log_path = os.path.join(output_dir, 'log.txt')
log_file = open(log_path, 'w', encoding='utf-8')
def log(msg):
    print(msg)
    log_file.write(msg + '\n')
# 添加 distance 列
step_km = 1  # 每个缓冲区之间的距离（单位：公里）
df['distance'] = (df['buffer_id'] - 30) * step_km
failed_cities = []

# 遍历每个城市
for city in df['city_name'].unique():
    df_city = df[df['city_name'] == city].sort_values('distance')

    try:
        # 提取变量
        distance = df_city['distance']
        bh_2010 = df_city['BH_2010']
        bh_2020 = df_city['BH_2020']
        lst_2010 = df_city['MODIS_2010']
        lst_2020 = df_city['MODIS_2020']
        predict_LST_2010 = df_city['predict_2010']
        predict_LST_2020 = df_city['predict_2020']

        # 插值前检查缺失
        df_valid = pd.DataFrame({
            'distance': distance,
            'bh_2010': bh_2010,
            'bh_2020': bh_2020,
            'lst_2010': lst_2010,
            'lst_2020': lst_2020,
            'predict_2010': predict_LST_2010,
            'predict_2020': predict_LST_2020
        }).replace([np.inf, -np.inf], np.nan).dropna()

        if len(df_valid) < 5:
            log(f"⚠️ {city} 建模预测失败，跳过绘图")
            failed_city.append((city, f"⚠️ {city} 建模预测失败，跳过绘图"))
            continue

        # 插值
        distance_smooth = np.linspace(df_valid['distance'].min(), df_valid['distance'].max(), 300)
        lst_2010_smooth = make_interp_spline(df_valid['distance'], df_valid['lst_2010'], k=3)(distance_smooth)
        lst_2020_smooth = make_interp_spline(df_valid['distance'], df_valid['lst_2020'], k=3)(distance_smooth)
        predict_LST_2010_smooth = make_interp_spline(df_valid['distance'], df_valid['predict_2010'], k=3)(
            distance_smooth)
        predict_LST_2020_smooth = make_interp_spline(df_valid['distance'], df_valid['predict_2020'], k=3)(
            distance_smooth)

        # 创建图
        fig, ax1 = plt.subplots(figsize=(8, 6))
        ax1.set_xlabel('Distance to Center (km)', fontsize=13)
        ax1.set_ylabel('Building Height (m)', fontsize=13)
        ax1.bar(distance, bh_2020, width=0.9 * step_km, color='#A9A9A9', label='BH_2020', alpha=0.8, zorder=1)
        ax1.bar(distance, bh_2010, width=0.9 * step_km, color='#8E8E8E', label='BH_2010', alpha=0.8, zorder=1)
        ax1.tick_params(axis='y')
        ax1.grid(True, which='both', linestyle='--', linewidth=0.7)

        ax2 = ax1.twinx()
        ax2.set_ylabel('LST (℃)', fontsize=13)
        ax2.plot(distance_smooth, lst_2020_smooth, color='#4EBEFF', linewidth=2, label='LST 2020 (℃)', zorder=3)
        ax2.plot(distance_smooth, lst_2010_smooth, color='#1F77B4', linewidth=2, label='LST 2010 (℃)', zorder=3)
        ax2.plot(distance_smooth, predict_LST_2020_smooth, color='#4EBEFF', linestyle='--', linewidth=2,
                 label='LST 2020 Predict (℃)', zorder=3, alpha=0.7)
        ax2.plot(distance_smooth, predict_LST_2010_smooth, color='#1F77B4', linestyle='--', linewidth=2,
                 label='LST 2010 Predict (℃)', zorder=3, alpha=0.7)
        ax2.plot(distance, lst_2020, 'o', color='#4EBEFF', markersize=3, zorder=4)
        ax2.plot(distance, lst_2010, 's', color='#1F77B4', markersize=3, zorder=4)
        ax2.plot(distance, predict_LST_2020, 'o', color='#4EBEFF', markersize=3, zorder=4, alpha=0.7)
        ax2.plot(distance, predict_LST_2010, 's', color='#1F77B4', markersize=3, zorder=4, alpha=0.7)
        ax2.tick_params(axis='y')

        # 图例合并
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        fig.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(0, 1),
                   bbox_transform=ax1.transAxes, fontsize=9)
        # 标题
        ax1.set_title(f'LST(MODIS) and Building Heights Along Distance to City Strip (3km interval)', fontsize=14)

        # 保存
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"{city}_BH_LST.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        log(f"✅ 已保存图像：{save_path}")

    except Exception as e:
     failed_city.append((city, str(e)))
     print(f"❌ {city} 出图失败，错误信息：{e}")
log_file.close()