import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.use('TkAgg')
from tqdm import tqdm

# 设置字体
plt.rcParams['font.sans-serif'] = ['Times New Roman']

def rose(df, city_name, dir_out, fig_name, rad=None):
    '''
    df 为 dataframe，行表示风向，列为风向频率
    rad 用于设置半径网格的刻度
    '''
    winds = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
             'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    angles = np.linspace(start=0, stop=360, num=17)

    # 读取数据列
    wind_direction = df['wind dominant direction']
    frequency = df['frequency']

    # 16方位制编号转角度
    direction_to_angle = {
        1: 0, 2: 22.5, 3: 45, 4: 67.5, 5: 90,
        6: 112.5, 7: 135, 8: 157.5, 9: 180, 10: 202.5,
        11: 225, 12: 247.5, 13: 270, 14: 292.5, 15: 315, 16: 337.5
    }

    wind_direction_angles = wind_direction.map(direction_to_angle)
    draw_data = np.array([wind_direction_angles, frequency])
    theta = draw_data[0] * np.pi / 180.

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='polar')

    ax.bar(theta, draw_data[1], width=0.4, bottom=0, color='blue', alpha=0.6, edgecolor='black')
    ax.set_thetagrids(angles[:-1], labels=winds, fontsize=15)
    ax.grid(lw=1, ls='--', color='black')
    ax.spines['polar'].set_linewidth(2)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_title(f"Wind Frequency Rose (2010_summer)", loc='center', fontsize=20)

    if rad:
        ax.set_rgrids(rad, angle=30, fontsize=15)

    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    plt.savefig(os.path.join(dir_out, fig_name), dpi=600)
    plt.close()

# 文件夹路径
csv_dir = r"E:\UHA\wind rose-N\2010_678_all_cities\frequency"
output_dir = r"E:\UHA\wind rose-N\2010_678_all_cities"

# 遍历所有csv文件，加上进度条
csv_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]

for filename in tqdm(csv_files, desc="正在生成玫瑰图"):
    filepath = os.path.join(csv_dir, filename)
    df = pd.read_csv(filepath)

    try:
        city_name = filename.split('_')[2]
    except IndexError:
        print(f"文件名格式错误：{filename}")
        continue

    fig_name = f"{city_name}夏季风向频率玫瑰图.png"
    rose(df, city_name, output_dir, fig_name, rad=[10, 20, 30, 40])
