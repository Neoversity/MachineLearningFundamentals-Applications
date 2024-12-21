# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 19:52:06 2024

@author: anton
"""

import os

# Змінюємо робочу директорію на 'final_proj'
os.chdir(r"C:\Users\anton\PythonProjects\Neoversity\machine_learning_fundamentals_and_applications\machine_learning_fundamentals_and_applications\nw_12")

# Перевіряємо, чи змінилася директорія
print("Current working directory:", os.getcwd())
#%%
# Завантаження набору даних
import pandas as pd
data = pd.read_pickle('mod_05_topic_10_various_data.pkl')
print(data.keys())
#%%
concrete_data = data['concrete']
concrete_data.head()
#%%
#Крок2 Використайте прийом підрахунку кількості для створення нової ознаки Components, яка вказуватиме на кількість задіяних складових у різних рецептурах бетону.
# Визначаємо стовпці, які стосуються компонентів бетону
components_columns = ['Cement', 'BlastFurnaceSlag', 'FlyAsh', 'Water', 
                      'Superplasticizer', 'CoarseAggregate', 'FineAggregate']

# Створюємо нову ознаку 'Components', що рахує кількість ненульових компонентів у кожному рядку
concrete_data['Components'] = concrete_data[components_columns].gt(0).sum(axis=1)

# Відображаємо перші кілька рядків, щоб перевірити нову ознаку
concrete_data[['Components']].head()
#%%
#Крок3. Нормалізуйте набір даних за допомогою об’єкта StandardScaler з пакета sklearn для подальшої кластеризації.
import numpy as np
from sklearn.preprocessing import StandardScaler

# Створюємо об'єкт StandardScaler
scaler = StandardScaler()

# Масштабуємо всі числові стовпці, включаючи CompressiveStrength
numeric_columns = concrete_data.select_dtypes(include=[np.number]).columns
normalized_data = scaler.fit_transform(concrete_data[numeric_columns])

# Перетворюємо назад у DataFrame для зручності
normalized_df = pd.DataFrame(normalized_data, columns=numeric_columns)

# Перевіряємо результат
normalized_df.head()

normalized_df.head()
#%%
### Крок4. Визначте оптимальну кількість кластерів за допомогою об'єкта KElbowVisualizer з пакета yellowbrick.
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os
os.environ["OMP_NUM_THREADS"] = "1"  # Обмежує кількість потоків
#%%
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

# Створюємо модель KMeans
kmeans = KMeans(random_state=42)

# Візуалізація оптимальної кількості кластерів за допомогою KElbowVisualizer
visualizer = KElbowVisualizer(kmeans, k=(2, 10), timings=False)
visualizer.fit(normalized_data)
visualizer.show()
#%%
### Крок 5. Проведіть кластеризацію методом k-середніх і отримайте мітки для кількості кластерів, визначеної на попередньому кроці.
# Створюємо модель KMeans з кількістю кластерів, визначеною на попередньому кроці
kmeans_final = KMeans(n_clusters=5, random_state=42)

# Навчаємо модель на нормалізованих даних
kmeans_final.fit(normalized_data)

# Отримуємо мітки кластерів
concrete_data['Cluster'] = kmeans_final.labels_

# Переглядаємо перші кілька рядків із мітками кластерів
concrete_data[['Cluster']].head()
#%%
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Виконуємо зменшення розмірності до 2 компонентів для візуалізації
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(normalized_data)

# Додаємо координати після PCA до набору даних
concrete_data['PCA1'] = reduced_data[:, 0]
concrete_data['PCA2'] = reduced_data[:, 1]

# Візуалізація кластерів
plt.figure(figsize=(10, 6))
for cluster in concrete_data['Cluster'].unique():
    cluster_data = concrete_data[concrete_data['Cluster'] == cluster]
    plt.scatter(cluster_data['PCA1'], cluster_data['PCA2'], label=f'Cluster {cluster}', alpha=0.7)

# Наносимо центроїди кластерів
centroids = pca.transform(kmeans_final.cluster_centers_)
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='black', marker='X', label='Centroids')

# Налаштування графіка
plt.title('Visualization of Clusters in 2D Space')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend()
plt.grid(True)
plt.show()
#%%
### Крок6. Використайте оригінальний набір вхідних даних для розрахунку описової статистики кластерів («звіту»)
# Розрахунок описової статистики (медіани) для кожного кластеру
cluster_statistics = concrete_data.groupby('Cluster').median()

# Додавання кількості компонент до описової статистики
cluster_statistics['Components_Count'] = concrete_data.groupby('Cluster')['Components'].sum()

# Виведення результату
cluster_statistics
#%%
### Крок7. Додайте до звіту кількість об'єктів (рецептур) у кожному з кластерів.
# Додаємо кількість об'єктів (рецептур) у кожному кластері до звіту
cluster_statistics['Recipe_Count'] = concrete_data.groupby('Cluster').size()

cluster_statistics.drop(columns=['Components_Count'], inplace=True)


# Виведення оновленого звіту
cluster_statistics
#%%
### Крок8. Проаналізуйте звіт та зробіть висновки.
#### Загальний висновок:
#- Кластери демонструють чітке групування за складом рецептури та міцністю бетону.
#- Кластер 1 — найбільш поширений і збалансований.
#- Кластер 2 — найменш міцний, орієнтований на економію цементу.
#- Кластер 4 — має найпростішу рецептуру, сфокусовану на використанні цементу.
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
import numpy as np

# Створюємо об'єкти для різних методів кластеризації
hac = AgglomerativeClustering(n_clusters=5)
gmm = GaussianMixture(n_components=5, random_state=42)
dbscan = DBSCAN(eps=0.5, min_samples=5)

# Виконуємо кластеризацію
hac_labels = hac.fit_predict(reduced_data)
gmm_labels = gmm.fit_predict(reduced_data)
dbscan_labels = dbscan.fit_predict(reduced_data)

# Готуємо дані для візуалізації
methods = {
    "DBSCAN": dbscan_labels,
    "kMeans": concrete_data['Cluster'],
    "HAC": hac_labels,
    "GMM": gmm_labels,
}

# Побудова графіків
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for ax, (method_name, labels) in zip(axes, methods.items()):
    scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='rainbow', alpha=0.7)
    ax.set_title(method_name)
    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    ax.grid(True)

# Налаштування загальної легенди
handles, labels = scatter.legend_elements()
fig.legend(handles, labels, title="Кластери", loc="upper right")

plt.tight_layout()
plt.show()
#%%
#kMeans:

#Кластери чітко розділені, особливо якщо дані мають симетричну форму. Підходить для сферичних кластерів.
#HAC (ієрархічна кластеризація):

#Дає схожий результат до kMeans, але з невеликими відмінностями через використання іншої метрики. Може краще працювати зі складними структурами.
#GMM (Gaussian Mixture Model):

#Добре працює з еліптичними кластерами, враховуючи ймовірнісну модель. Враховує більше варіацій, ніж kMeans.
#DBSCAN:

#Виділяє області з високою щільністю. Кластер -1 представляє шумові точки, які не входять до жодного кластера.