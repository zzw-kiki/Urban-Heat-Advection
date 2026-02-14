import pandas as pd

# 读取 .dta 文件
dta_file = r"E:\UHAE\1980-01-01~2024-08-01中国各省市区县日度主导风向及平均风向计算结果\1980-01-01~2024-08-01中国各城市日度主导风向及平均风向计算结果.dta"
df = pd.read_stata(dta_file)

# 确保 date 列是日期格式
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# 过滤出 2020 年的数据
df_2020 = df[df['date'].dt.year == 2010]
df_2020_sumer_678 = df[(df['date'].dt.year == 2010) & (df['date'].dt.month.isin([6, 7,8]))]
'''
# 过滤出市列为指定城市的数据
cities = ['南京市', '上海市', '武汉市', '北京市', '广州市', '深圳市', '成都市', '重庆市', '苏州市', '杭州市','大连市','西安市','石家庄市',
          '济南市','贵阳市','长春市','厦门市','太原市','南昌市','佛山市','昆明市','沈阳市','长沙市','海口市','青岛市','哈尔滨市','郑州市','无锡市','乌鲁木齐市',
          '天津市','福州市','宁波市','合肥市']
df_filtered = df_2020[df_2020['市'].isin(cities)]
'''
# 保存为 CSV 文件，指定 UTF-8 编码
csv_file = r"E:\UHAE\1980-01-01~2024-08-01中国各省市区县日度主导风向及平均风向计算结果\2010_678中国各城市主导风向及平均风向.csv"
df_2020_sumer_678.to_csv(csv_file, index=False, encoding='utf-8-sig')  # 使用 UTF-8 编码

print(f"2020年十大城市数据已成功保存为 CSV 格式: {csv_file}")
