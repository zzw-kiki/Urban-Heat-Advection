import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
plt.rcParams["font.family"] = "Times New Roman"
# 读取数据
file = r"E:\UHA\final.xlsx"
df = pd.read_excel(file)

# 选出 GDP 和 population 都不为空的数据

df['population'] = df['population'] / 10000
valid_data = df[['GDP', 'population']].dropna()
# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(valid_data)

# KMeans 聚类 (3类)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

# 在原始表中新建 Cluster 列，先设为空
df['Cluster'] = pd.NA

# 把聚类结果填回到非空的行
df.loc[valid_data.index, 'Cluster'] = labels

# 计算聚类效果指标
silhouette = silhouette_score(X_scaled, labels)
ch = calinski_harabasz_score(X_scaled, labels)
db = davies_bouldin_score(X_scaled, labels)

print("聚类效果指标：")
print(f"  轮廓系数 (Silhouette Score): {silhouette:.3f}")
print(f"  Calinski-Harabasz 指数: {ch:.3f}")
print(f"  Davies-Bouldin 指数: {db:.3f}")

# 聚类结果可视化

# 给 3 类指定颜色（比如红、蓝、绿）
colors = {0: "#440154", 1: "#FF1C4D", 2: "#63B1AE", 3:"#FF7E52"}
#FF7E52
plt.figure(figsize=(8,6))
for cluster in range(4):
    cluster_data = valid_data[labels == cluster]
    plt.scatter(
        cluster_data['GDP'],
        cluster_data['population'],
        c=colors[cluster],
        label=f"Cluster {cluster}",
        s=70,
        alpha=0.6
    )

# 画聚类中心（反标准化）
centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centers[:,0], centers[:,1], c='black', s=60, marker='x', label='Centers')

plt.xlabel('GDP (billion yuan)',fontsize=12)
plt.ylabel('Population (ten thousand persons)',fontsize=12)
plt.title('K-means Result',fontsize=12)
plt.legend(loc='upper left')
plt.grid(True)
# plt.savefig(r"E:\WBTI\筛选结果\论文\论文出图\K-means.png",dpi=600)
plt.show()

# 按类别统计平均GDP和平均人口
cluster_summary = df.dropna(subset=['Cluster']).groupby('Cluster')[['GDP', 'population']].mean().round(2)

print("\n每一类城市的平均特征：")
print(cluster_summary)
mapping = {
    0: 'small',
    1: 'Mega',
    2: 'middle',
    3: 'large'
}
df['size_2'] = df['Cluster'].map(mapping)

# 保存结果（可选）
out_path = r"E:\WBTI\筛选结果\论文\final_2_with_cluster.xlsx"
#df.to_excel(out_path, index=False)
#summary_path = r"E:\WBTI\筛选结果\论文\cluster_summary.xlsx"
#cluster_summary.to_excel(summary_path)
