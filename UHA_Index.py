import pandas as pd
import numpy as np
import os
# 最终指标不适用sigmod函数，sigmod值>0.5对应残差>0
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_uha(df, k=10):
    all_results = []

    for city in df['city_name'].unique():
        df_city = df[df['city_name'] == city].copy()
        # 检查是否存在 distance == 0 的中心点数据
        '''
        df_center = df_city[df_city['distance'] == 0]
        if df_center.empty:
            print(f"[跳过] 城市 {city} 缺失中心点（bufferid == 30）数据")
            continue  # 跳过本城市
        '''
        # 城市中心温度
        # T0_2010 = df_city[df_city['distance'] == 0]['MODIS_2010'].values[0]
        # T0_2020 = df_city[df_city['distance'] == 0]['MODIS_2020'].values[0]

        # ΔT 和 ΔΔT
        # df_city['delta_T_2010'] = df_city['residual_2010'] - T0_2010
        df_city['residual_2010'] = df_city['residual_2010']
        # df_city['delta_T_2010'] = df_city['MODIS_2010'] -T0_2010
        # df_city['delta_T_2020'] = df_city['residual_2020'] - T0_2020
        df_city['residual_2020'] = df_city['residual_2020']
        # df_city['delta_T_2020'] = df_city['MODIS_2020'] - T0_2020 - df_city['predict_20-10']

        df_city['delta_residual'] = df_city['residual_2020'] - df_city['residual_2010']
        df_downwind = df_city[df_city['distance'] > 0].copy()
        df_upwind = df_city[df_city['distance'] < 0].copy()
        # print(f"{city} 最远距离：", df_downwind['distance'].max())

        def compute_w(row):
            return sigmoid(k * row['delta_residual'])
        '''
        def compute_w(row):
            if row['delta_T_2010'] <= 0:
                return sigmoid(k * row['delta_delta_T'])
            elif row['delta_T_2010'] > 0:
                return sigmoid(k * row['delta_delta_T'])
        '''

        df_downwind['UHA_value'] = df_downwind.apply(compute_w, axis=1)

        # 构建以城市为行，UHA_distance为列的结构
        row = {'city': city}
        uha_columns = []
        for _, r in df_downwind.iterrows():
            col_name = f"UHA_{int(r['distance'])}"
            row[col_name] = r['UHA_value']
            uha_columns.append(col_name)

        # row['AVG_WBTI_all'] = df_downwind['WBTI_value'].mean()

        # ΔT计算：下风区 - 上风区的平均温度差
        for dist_limit in range(1, 31):
            # 下风口 0 < d ≤ dist_limit
            dw_2020 = df_downwind[(df_downwind['distance'] > 0) & (df_downwind['distance'] <= dist_limit)]['MODIS_2020']
            uw_2020 = df_upwind[(df_upwind['distance'] < 0) & (df_upwind['distance'] >= -dist_limit)]['MODIS_2020']

            dw_2010 = df_downwind[(df_downwind['distance'] > 0) & (df_downwind['distance'] <= dist_limit)]['MODIS_2010']
            uw_2010 = df_upwind[(df_upwind['distance'] < 0) & (df_upwind['distance'] >= -dist_limit)]['MODIS_2010']

            if not dw_2020.empty and not uw_2020.empty:
                row[f'ΔT2020_{dist_limit}km'] = dw_2020.mean() - uw_2020.mean()
            else:
                row[f'ΔT2020_{dist_limit}km'] = np.nan

            if not dw_2010.empty and not uw_2010.empty:
                row[f'ΔT2010_{dist_limit}km'] = dw_2010.mean() - uw_2010.mean()
            else:
                row[f'ΔT2010_{dist_limit}km'] = np.nan


        # 累进平均值输出：1~1km，1~2km，…，1~30km
        for upper in range(1, 31):  # 从1到30（包含30）
            avg_col = f'AVG_UHA_1_{upper}'
            subset = df_downwind[(df_downwind['distance'] >= 1) & (df_downwind['distance'] <= upper)]
            if not subset.empty:
                row[f'AVG_UHA_1_{upper}'] = subset['UHA_value'].mean()
            else:
                row[f'AVG_UHA_1_{upper}'] = np.nan
            uha_columns.append(avg_col)

        # 最大WBTI及其对应距离
        if uha_columns:
            uha_vals = {col: row[col] for col in uha_columns}
            max_col = max(uha_vals, key=uha_vals.get)
            row['Max_UHA_column'] = max_col
            row['Max_UHA_value'] = uha_vals[max_col]
        else:
            row['Max_UHA_column'] = np.nan
            row['Max_UHA_value'] = np.nan

        # 添加下风口与上风口温差最大的距离
        # 找出 ΔT2010 和 ΔT2020 最大值及其对应列名
        dt2010_vals = {k: v for k, v in row.items() if k.startswith('ΔT2010_')}
        dt2020_vals = {k: v for k, v in row.items() if k.startswith('ΔT2020_')}

        if dt2010_vals:
            dt2010_max_col = max(dt2010_vals, key=dt2010_vals.get)
            row['ΔT2010_MAX_column'] = dt2010_max_col
            row['ΔT2010_MAX_value'] = dt2010_vals[dt2010_max_col]
        else:
            row['ΔT2010_MAX_column'] = np.nan
            row['ΔT2010_MAX_value'] = np.nan

        if dt2020_vals:
            dt2020_max_col = max(dt2020_vals, key=dt2020_vals.get)
            row['ΔT2020_MAX_column'] = dt2020_max_col
            row['ΔT2020_MAX_value'] = dt2020_vals[dt2020_max_col]
        else:
            row['ΔT2020_MAX_column'] = np.nan
            row['ΔT2020_MAX_value'] = np.nan

        all_results.append(row)

    '''
    # 转换为 DataFrame
    df_result = pd.DataFrame(all_results)
    df_result = df_result.set_index('city')
    df_result = df_result.sort_index(axis=1)  # 按列名排序
    return df_result.reset_index()
    '''
    df_result = pd.DataFrame(all_results)

    import re

    def extract_number(colname, prefix):
        # 从colname去掉prefix后提取数字（带km也去掉）
        s = colname.replace(prefix, '').replace('km', '')
        return int(re.findall(r'\d+', s)[0]) if re.findall(r'\d+', s) else float('inf')

    # 各类别列
    avg_cols = sorted(
        [col for col in df_result.columns if col.startswith('AVG_UHA_1_')],
        key=lambda x: extract_number(x, 'AVG_UHA_1_')
    )

    dt2020_cols = sorted(
        [col for col in df_result.columns if col.startswith('ΔT2020_') and 'MAX' not in col],
        key=lambda x: extract_number(x, 'ΔT2020_')
    )

    dt2010_cols = sorted(
        [col for col in df_result.columns if col.startswith('ΔT2010_') and 'MAX' not in col],
        key=lambda x: extract_number(x, 'ΔT2010_')
    )

    max_cols = ['Max_UHA_column', 'Max_UHA_value', 'ΔT2010_MAX_column', 'ΔT2010_MAX_value', 'ΔT2020_MAX_column', 'ΔT2020_MAX_value']

    uha_cols = [col for col in df_result.columns if col.startswith('UHA_') and not col.startswith('AVG_UHA_1_')]


    # WBTI 升序排列（1 到 30）
    uha_cols_sorted = sorted(uha_cols, key=lambda x: extract_number(x, 'UHA_'))

    # 拼接最终列名
    ordered_cols = (
            ['city'] +
            avg_cols +
            uha_cols_sorted +
            max_cols +
            dt2020_cols +
            dt2010_cols


    )

    # 重新排序
    df_result = df_result[ordered_cols]

    return df_result


# 读取数据
input_path = r"E:\UHA\regression\3km_1km_buffer_center\regression_CPR.csv"
df = pd.read_csv(input_path, encoding='utf-8')

# 循环计算不同 k 值
output_base_path = r"E:\UHA\UHA_result\3km_1km_center"
all_counts_list = []  # 存放每个k的统计结果
for k in [1, 5, 10, 15, 20]:
    print(f"正在计算 k = {k} 的结果...")
    result_df = compute_uha(df, k=k)

    # 保存主WBTI结果
    result_path = os.path.join(output_base_path, f"UHA_3km_1km_k{k}.csv")
    result_df.to_csv(result_path, index=False, encoding='utf-8-sig')
    print(f"已保存 k = {k} 的 UHA 表格：{result_path}")


    # 统计大于 0.5 的城市数量
    uha_cols = [col for col in result_df.columns if col.startswith("WBTI_") or col.startswith("AVG_WBTI_")]
    uha_counts = (result_df[wbti_cols] > 0.5).sum(axis=0)
    uha_counts.name = f'num_cities_over_0.5_k={k}'  # 列名添加 k 标记
    all_counts_list.append(wbti_counts)

    # 合并所有k列（按列方向）
    combined_counts_df = pd.concat(all_counts_list, axis=1).reset_index()
    combined_counts_df.rename(columns={'index': 'UHA_column'}, inplace=True)

    # 保存为CSV
combined_path = os.path.join(output_base_path, "UHA_over_0.5_city_count.csv")
combined_counts_df.to_csv(combined_path, index=False, encoding='utf-8-sig')
print("所有k值的统计结果已按列合并保存：", combined_path)
