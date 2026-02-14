import pandas as pd

# ================= 文件路径 =================
base_file = r"E:\UHAE\socio-economic-data\251_cities.xlsx"
gdp_file = r"E:\UHAE\socio-economic-data\GDP.xlsx"
pop_file = r"E:\UHAE\socio-economic-data\population.csv"

out_file = r"E:\UHAE\socio-economic-data\251_cities_with_GDP_Pop.xlsx"

# ================= 读取基础城市表 =================
df_base = pd.read_excel(base_file)

# ================= 读取并处理人口数据 =================
df_pop = pd.read_csv(pop_file)

# 如果 population.csv 只包含 2020 年，可直接选列
df_pop_2020 = df_pop[['city_name', 'SUM']]

# 合并人口
df_merged = df_base.merge(
    df_pop_2020,
    left_on='city',
    right_on='city_name',
    how='left'
)

df_merged = df_merged.drop(columns=['city_name']) \
                     .rename(columns={'SUM': 'population'})

# ================= 读取并处理 GDP 数据 =================
df_gdp = pd.read_excel(gdp_file)

# 筛选 2020 年 GDP
df_gdp_2020 = df_gdp[df_gdp['年份'] == 2020][['地区名称', 'GDP(亿元)']]

# 合并 GDP
df_merged = df_merged.merge(
    df_gdp_2020,
    left_on='city',
    right_on='地区名称',
    how='left'
)

df_merged = df_merged.drop(columns=['地区名称']) \
                     .rename(columns={'GDP(亿元)': 'GDP'})

# ================= 保存结果 =================
df_merged.to_excel(out_file, index=False)

print("已成功保存：", out_file)
