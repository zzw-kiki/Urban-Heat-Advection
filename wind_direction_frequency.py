import pandas as pd

# 读取 2020 年十大城市日度主导风向及平均风向计算结果 CSV 文件
csv_file = r"E:\UHA\1980-01-01~2024-08-01中国各省市区县日度主导风向及平均风向计算结果\2010_678中国各城市主导风向及平均风向.csv"
df = pd.read_csv(csv_file, encoding='utf-8-sig')

# 确保 date 列是日期格式
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# 过滤出 2020 年的数据
df_2020 = df[df['date'].dt.year == 2010]
# df_august_2020 = df[(df['date'].dt.year == 2020) & (df['date'].dt.month == 8)]
df_2020_summer_678 = df[(df['date'].dt.year == 2010) & (df['date'].dt.month.isin([6, 7,8]))]
'''
# 需要处理的城市列表
cities = ['南京市', '上海市', '武汉市', '北京市', '广州市', '深圳市', '成都市', '重庆市', '苏州市', '杭州市','大连市','西安市','石家庄市',
          '济南市','贵阳市','长春市','厦门市','太原市','南昌市','佛山市','昆明市','沈阳市','长沙市','海口市','青岛市','哈尔滨市','郑州市','无锡市','乌鲁木齐市',
          '天津市','福州市','宁波市','合肥市']
'''
# 获取所有城市名称（去重）
cities = df_2020_summer_678['市'].dropna().unique()
txt_output = r"E:\WBTI\wind rose-N\2010_678_all_cities\frequency\city_list_2010.txt"
with open(txt_output, 'w', encoding='utf-8') as f:
    for city in cities:
        f.write(city + '\n')
print(f"城市列表已保存为 TXT 文件：{txt_output}")
# 遍历每个城市进行统计并输出
for city in cities:
    df_city = df_2020_summer_678[df_2020_summer_678['市'] == city]

    # 计算主导风向频率
    wind_direction_frequency = df_city['dominant_wind_direction'].value_counts().sort_index()
    wind_direction_df = wind_direction_frequency.reset_index()
    wind_direction_df.columns = ['wind dominant direction', 'frequency']

    # 输出路径
    csv_output = f"E:\\WBTI\\wind rose-N\\2010_678_all_cities\\frequency\\2010_678_{city}_风向频率玫瑰图.csv"
    wind_direction_df.to_csv(csv_output, index=False, encoding='utf-8-sig')

    print(f"{city} 风向频率数据已成功保存为 CSV 格式: {csv_output}")
