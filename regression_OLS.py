import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import statsmodels.api as sm
import os
import matplotlib
matplotlib.use('TkAgg')
# 每个城市单独训练一个模型
failed_models = []  # 存储未建模成功的城市-年份
model_stats = []  # 存储每个城市-年份的R, R², p值
input_path = r"E:\UHAE\wind_path_statistic\3km_1km_buffer\data.csv"
output_csv = r"E:\UHAE\regression\3km_1km_buffer\regression_OLS.csv" # 模型回归结果
plot_dir = r"E:\UHAE\regression\3km_1km_buffer\regression_result_OLS"
stats_csv_path = r"E:\UHAE\regression\3km_1km_buffer\model_stats_OLS.csv" # 模型统计评价指标结果

os.makedirs(plot_dir, exist_ok=True)
log_path = os.path.join(plot_dir, 'log.txt')
log_file = open(log_path, 'w', encoding='utf-8')
def log(msg):
    print(msg)
    log_file.write(msg + '\n')
# 读取数据
df = pd.read_csv(input_path)
df['FID'] = df.index  # 添加唯一编号

def fit_model_city_year(df_city, city, year, residual_threshold=2.0):
    bh_col = f'BH_{year}'
    lst_col = f'MODIS_{year}'
    pred_col = f'predict_{year}'
    resid_col = f'residual_{year}'

    # 仅 distance<0 数据做训练
    df_train = df_city[df_city['distance'] <= 0].copy()
    X_raw = df_train[bh_col]
    y_raw = df_train[lst_col]

    if len(df_train) < 4 or X_raw.isnull().any() or y_raw.isnull().any():
        log(f"[跳过] {city} - {year} 数据不足或含NaN，无法建模。")
        failed_models.append((city, year))
        return df_city

    # 初次拟合
    X_const = sm.add_constant(X_raw)
    model_raw = sm.OLS(y_raw, X_const).fit()
    residuals = y_raw - model_raw.predict(X_const)

    # 标准化残差，剔除异常值
    z_scores = (residuals - residuals.mean()) / residuals.std()
    mask = z_scores.abs() <= residual_threshold
    df_train_clean = df_train[mask].copy()
    log(f"【{city} - {year}】剔除异常值前样本：{len(df_train)}, 剩余：{len(df_train_clean)}")

    if len(df_train_clean) < 5:
        log(f"[跳过] {city} - {year} 剔除后样本过少，无法建模")
        failed_models.append((city, year))
        return df_city

    # 再次拟合
    X_train = sm.add_constant(df_train_clean[bh_col])
    y_train = df_train_clean[lst_col]
    model = sm.OLS(y_train, X_train).fit()

    # 相关系数
    r_value, p_value = pearsonr(df_train_clean[bh_col], y_train)

    log(f"------ {city} - {year} 模型 ------")
    log(f"Pearson R: {r_value:.4f}, p: {p_value:.4e}, R²: {model.rsquared:.4f}")
    # 计算 RMSE（均方根误差）
    y_pred_train = model.predict(X_train)
    rmse = np.sqrt(np.mean((y_train - y_pred_train) ** 2))
    log(f"RMSE: {rmse:.4f}")

    # 保存模型统计指标
    model_stats.append({
        'city_name': city,
        'year': year,
        'R': round(r_value, 4),
        'R的平方': round(model.rsquared, 4),
        'p值': format(p_value, '.4e'),
        'RMSE': round(rmse, 4),
        '样本数量': len(df_train_clean)
    })

    '''
    # 全部点预测
    X_all = sm.add_constant(df_city[bh_col])
    df_city[pred_col] = model.predict(X_all)
    df_city[resid_col] = df_city[lst_col] - df_city[pred_col]
    '''
    # 只预测下风口区域
    df_predict = df_city[df_city['distance'] > 0].copy()
    # 如果样本不足，跳过预测
    if df_predict[bh_col].dropna().shape[0] < 2:
        log(f"[跳过预测] {city} - {year} 下风口数据<=1（{df_predict[bh_col].dropna().shape[0]}），无法预测。")
        failed_models.append((city, year))
        return df_city # 海口市下风口区域只有一个点，数学理论上可以算，但是代码期望输入的维度不是这个
    X_predict = sm.add_constant(df_predict[bh_col])
    df_city.loc[df_city['distance'] > 0, pred_col] = model.predict(X_predict)
        # 上风口区域（distance <= 0）使用原始值
    df_city.loc[df_city['distance'] <= 0, pred_col] = df_city.loc[df_city['distance'] <= 0, lst_col]
        # 计算残差（真实 - 预测）
    df_city[resid_col] = df_city[lst_col] - df_city[pred_col]


    # 可视化
    x_pred = np.linspace(df_train_clean[bh_col].min(), df_train_clean[bh_col].max(), 100)
    X_pred = sm.add_constant(x_pred)
    pred_summary = model.get_prediction(X_pred).summary_frame(alpha=0.05)

    plt.figure(figsize=(8, 6))
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.scatter(df_train_clean[bh_col], y_train, alpha=0.7, label='Observed', s=10)
    plt.plot(x_pred, pred_summary['mean'], color='red', label='Regression Line')
    plt.fill_between(x_pred, pred_summary['mean_ci_lower'], pred_summary['mean_ci_upper'],
                     color='red', alpha=0.2, label='95% Confidence Interval')

    plt.xlabel(f'{bh_col} (m)')
    plt.ylabel(f'{lst_col} (℃)')
    plt.title(f'Regression of {lst_col} vs {bh_col} ({year})')
    plt.legend(loc='upper left')

    text = (f"n_samples = {len(df_train_clean)}\n"
            f"p = {p_value:.4f}\n"
            f"R = {r_value:.4f}\n"
            f"R² = {model.rsquared:.4f}\n"
            f"RMSE = {rmse:.4f} ℃\n"
            f"y = {model.params.iloc[1]:.3f}x + {model.params.iloc[0]:.3f}")
    plt.text(0.95, 0.05, text, color='black', weight='bold',
             verticalalignment='bottom', horizontalalignment='right',
             transform=plt.gca().transAxes, fontsize=12)

    # 保存图像
    city_safe = city.replace('/', '_').replace('\\', '_')
    plot_path = os.path.join(plot_dir, f'{city_safe}_{year}_regression_2.0.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"图像已保存：{plot_path}\n")

    return df_city

# 年份列表
years = ['2010', '2020']

# 最终结果列表
df_all_results = []

# 按城市处理
for city, df_city in df.groupby('city_name'):
    log(f"====== 处理城市：{city} ======")
    for year in years:
        df_city = fit_model_city_year(df_city, city, year)
    df_all_results.append(df_city)

# 合并所有城市数据并保存
df_result = pd.concat(df_all_results, ignore_index=True)
df_result.to_csv(output_csv, index=False, encoding='utf-8-sig')
print(f"模型残差结果已保存至：{output_csv}")
# 保存模型统计指标结果
df_stats = pd.DataFrame(model_stats)

df_stats.to_csv(stats_csv_path, index=False, encoding='utf-8-sig')
print(f"模型统计指标已保存至：{stats_csv_path}")

if failed_models:
    log("\n以下城市-年份未能成功建模：")
    for city, year in failed_models:
        log(f"- {city} - {year}")
else:
    log("\n所有城市均成功建模。")
log_file.close()
