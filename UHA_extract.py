import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline

# 文件路径
uha_bh_path = r"E:\UHA\UHA_BH.csv"
regression_path = r"E:\UHA\regression\3km_1km_buffer_center\regression_CPR.csv"
output_path = r"E:\UHA\UHA_BH_updated.csv"

# 读取 CSV
uha_df = pd.read_csv(uha_bh_path)
reg_df = pd.read_csv(regression_path)

# 初始化新列
uha_df['BH_2010'] = np.nan
uha_df['BH_2020'] = np.nan
uha_df['range_2010'] = np.nan
uha_df['range_2020'] = np.nan
uha_df['temp_2010'] = np.nan
uha_df['temp_2020'] = np.nan


def calculate_positive_range_and_mean(dist_new, resid_new):
    """
    严格计算 residual>0 的连续区间总长度和平均值
    """
    positive_mask = resid_new > 0
    range_value = 0
    temp_values = []

    start_idx = None
    for i, val in enumerate(positive_mask):
        if val and start_idx is None:
            # 开始新的区间
            start_idx = i
        elif not val and start_idx is not None:
            # 区间结束
            length = dist_new[i - 1] - dist_new[start_idx]  # 末点减起点
            range_value += length
            temp_values.extend(resid_new[start_idx:i])
            start_idx = None

    # 如果最后一个点仍然是正值
    if start_idx is not None:
        length = dist_new[-1] - dist_new[start_idx]
        range_value += length
        temp_values.extend(resid_new[start_idx:])

    temp_mean = np.mean(temp_values) if temp_values else 0
    return range_value, temp_mean


# 遍历每个城市
for idx, row in uha_df.iterrows():
    city = row['city']

    # 筛选该城市的数据
    city_data = reg_df[reg_df['city_name'] == city]

    if city_data.empty:
        continue

    # 1. distance==0 的 BH 值
    bh0 = city_data[city_data['distance'] == 0]
    if not bh0.empty:
        uha_df.at[idx, 'BH_2010'] = bh0['BH_2010'].values[0]
        uha_df.at[idx, 'BH_2020'] = bh0['BH_2020'].values[0]

    # 2. distance > 0 的残差曲线
    for year in ['2010', '2020']:
        dist = city_data['distance'][city_data['distance'] >= 0].values
        resid = city_data[f'residual_{year}'][city_data['distance'] >= 0].values

        if len(dist) < 2:
            continue

        # 对 dist 排序，同时对 resid 排序
        sorted_idx = np.argsort(dist)
        dist_sorted = dist[sorted_idx]
        resid_sorted = resid[sorted_idx]

        # 样条插值生成300个点
        spline = UnivariateSpline(dist_sorted, resid_sorted, s=0)
        dist_new = np.linspace(dist_sorted.min(), dist_sorted.max(), 300)
        resid_new = spline(dist_new)

        # 严格计算连续区间
        range_value, temp_value = calculate_positive_range_and_mean(dist_new, resid_new)

        uha_df.at[idx, f'range_{year}'] = range_value
        uha_df.at[idx, f'temp_{year}'] = temp_value

# 保存结果
uha_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print("处理完成，结果已保存至:", output_path)
