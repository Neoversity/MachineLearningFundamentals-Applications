# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 22:13:52 2024

@author: anton
"""

import os

# Змінюємо робочу директорію на 'hw_8'
os.chdir(r"C:\Users\anton\PythonProjects\Neoversity\machine_learning_fundamentals_and_applications\machine_learning_fundamentals_and_applications\hw_10")

# Перевіряємо, чи змінилася директорія
print("Current working directory:", os.getcwd())
#%%
# Крок 1: Імпорт необхідних бібліотек
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
#%%
# Завантаження набору даних
data = pd.read_pickle('mod_05_topic_10_various_data.pkl')
# Витяг даних за ключем 'autos'
df = data['autos']

# Перевірка структури витягнутого DataFrame
print(df.head())
#%%
# Додавання нової ознаки 'stroke_ratio'
df['stroke_ratio'] = df['stroke'] / df['bore']

# Перевірка нової ознаки
print(df[['stroke', 'bore', 'stroke_ratio']].head())

#%%%
# Крок 2. Визначте перелік дискретних ознак (в широкому розумінні) для подальшого розрахунку показника взаємної інформації.
print(df.dtypes)
# Виявлення дискретних ознак
discrete_features = df.select_dtypes(include=['object', 'int']).columns.tolist()

print("Дискретні ознаки:")
print(discrete_features)
#%%
# Виведення таблиці в стандартному вигляді
discrete_features = ['make', 'fuel_type', 'aspiration', 'num_of_doors', 'body_style',
                     'drive_wheels', 'engine_location', 'curb_weight', 'engine_type',
                     'num_of_cylinders', 'engine_size', 'fuel_system',
                     'compression_ratio', 'horsepower', 'peak_rpm', 'city_mpg',
                     'highway_mpg', 'price']

# Створення DataFrame
discrete_features_df = pd.DataFrame({'Discrete Features': discrete_features})

# Виведення таблиці
print(discrete_features_df)
#%%
# Крок 3
# Перетворення категоріальних ознак у числовий формат
X = df.drop(columns=['price'])  # Всі вхідні ознаки
y = df['price']  # Цільова змінна

# Використання One-Hot Encoding для категоріальних змінних
X = pd.get_dummies(X, drop_first=False)  # Не використовуємо drop_first, щоб врахувати всі категорії

# Обчислення взаємної інформації
mi_scores = mutual_info_regression(X, y)

# Створення таблиці з результатами
mi_scores_df = pd.DataFrame({
    'Feature': X.columns,
    'Mutual Information Score': mi_scores
}).sort_values(by='Mutual Information Score', ascending=False)

# Виведення результатів
print(mi_scores_df)
#%%%
# Побудова графіка
mi_scores_df.plot(x='Feature', y='Mutual Information Score', kind='barh')
plt.title("Mutual Information Scores")
plt.xlabel("Score")
plt.ylabel("Features")
plt.gca().invert_yaxis()  # Інвертуємо осі для кращого вигляду
plt.show()
#%%
# Крок 4
# Копія даних для підготовки
X = df.drop(columns=['price'])
y = df['price']

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(drop=None, sparse_output=False)
X_encoded = ohe.fit_transform(X.select_dtypes(include=['object']))

# Замінюємо категоріальні колонки в X на закодовані
encoded_columns = ohe.get_feature_names_out(X.select_dtypes(include=['object']).columns)
X = pd.concat(
    [pd.DataFrame(X_encoded, columns=encoded_columns), X.select_dtypes(exclude=['object'])],
    axis=1
)


# Поділ даних на тренувальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#%%
# Ініціалізація та навчання моделі
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Оцінка важливості ознак
feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)

# Виведення результатів
print("Feature Importance:")
print(feature_importance)
#%%
# Візуалізація важливості ознак
feature_importance.plot(kind='barh', figsize=(10, 8))
plt.title("Feature Importance in RandomForestRegressor")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.gca().invert_yaxis()  
plt.show()
#%%
# Ініціалізація та навчання моделі
gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_train, y_train)

# Оцінка важливості ознак
gb_feature_importance = pd.Series(gb_model.feature_importances_, index=X.columns).sort_values(ascending=False)

# Виведення результатів
print("Feature Importance (GradientBoostingRegressor):")
print(gb_feature_importance)
#%%
# Візуалізація важливості ознак
gb_feature_importance.plot(kind='barh', figsize=(10, 8))
plt.title("Feature Importance in GradientBoostingRegressor")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.gca().invert_yaxis()  # Інвертуємо осі для кращого вигляду
plt.show()
#%%


#%%
# Крок 5
# Перевірка розмірів даних
print(f"Shapes of inputs:")
print(f"mi_scores_df: {mi_scores_df.shape}")
print(f"feature_importance: {feature_importance.shape}")
print(f"gb_feature_importance: {gb_feature_importance.shape}")
#%%

# Перевірка індексів та значень
print("Індекси mi_scores_df:")
print(mi_scores_df.index)
print("Значення стовпця Feature у mi_scores_df:")
print(mi_scores_df['Feature'].head())

print("Індекси feature_importance:")
print(feature_importance.index)
print("Значення feature_importance:")
print(feature_importance.head())

print("Індекси gb_feature_importance:")
print(gb_feature_importance.index)
print("Значення gb_feature_importance:")
print(gb_feature_importance.head())
#%%
# Об'єднання даних за ознаками
comparison_df = mi_scores_df[['Feature', 'Mutual Information Score']].copy()

# Додаємо Random Forest і Gradient Boosting важливості за назвами ознак
comparison_df = comparison_df.merge(
    feature_importance.rename('Random Forest Importance'),
    left_on='Feature', right_index=True
)

comparison_df = comparison_df.merge(
    gb_feature_importance.rename('Gradient Boosting Importance'),
    left_on='Feature', right_index=True
)

# Масштабування взаємної інформації
comparison_df['Mutual Information Score'] /= comparison_df['Mutual Information Score'].sum()

# Додавання ранжування
comparison_df['MI Rank'] = comparison_df['Mutual Information Score'].rank(pct=True, method='average')
comparison_df['RF Rank'] = comparison_df['Random Forest Importance'].rank(pct=True, method='average')
comparison_df['GB Rank'] = comparison_df['Gradient Boosting Importance'].rank(pct=True, method='average')


# Виведення результатів
print(comparison_df)
#%%
# Виведення всіх стовпців
pd.set_option('display.max_columns', None)

# Перевіряємо результат
print(comparison_df)
#%%
# Крок 6
# Переформатування даних для абсолютних значень
abs_melted_df = comparison_df.reset_index().melt(
    id_vars="Feature",  # Переконайся, що Feature є стовпцем
    value_vars=["Mutual Information Score", "Random Forest Importance", "Gradient Boosting Importance"],
    var_name="Method",
    value_name="Importance"
)

# Побудова графіка для абсолютних значень
sns.set(style="whitegrid")
plt.figure(figsize=(14, 10))
sns.barplot(
    data=abs_melted_df,
    x="Importance",
    y="Feature",
    hue="Method",
    palette="muted",
    orient="h"
)

plt.title("Comparison of Feature Importances (Absolute Values)", fontsize=14)
plt.xlabel("Absolute Importance", fontsize=12)
plt.ylabel("Features", fontsize=12)
plt.legend(title="Method", fontsize=10, loc="lower right")
plt.tight_layout()
plt.show()

#%%
print(comparison_df[['Mutual Information Score', 'Random Forest Importance', 'Gradient Boosting Importance']])
#%%
# Підготовка даних у форматі для catplot
melted_df = comparison_df.melt(
    id_vars="Feature",
    value_vars=["MI Rank", "RF Rank", "GB Rank"],
    var_name="Method",
    value_name="Rank"
)

# Перевірка структури
print(melted_df.head())

# Побудова згрупованого barplot
sns.set_theme(style="whitegrid")
g = sns.catplot(
    data=melted_df,
    kind="bar",
    x="Rank",
    y="Feature",
    hue="Method",
    palette="dark",
    alpha=.6,
    height=10,  # Для більшої висоти графіка
    orient="h"
)
g.despine(left=True)
g.set_axis_labels("Rank (scaled to 0 to 1)", "Feature")
g.legend.set_title("")
plt.title("Grouped Barplots of Feature Ranks Across Methods (Horizontal)")
plt.tight_layout()
plt.show()
#%%
print(comparison_df[["MI Rank", "RF Rank", "GB Rank"]])
#%%
correlation_matrix = comparison_df[["Mutual Information Score", "Random Forest Importance", "Gradient Boosting Importance"]].corr()
print(correlation_matrix)
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix of Feature Importance Metrics")
plt.show()