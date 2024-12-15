# Импортируем необходимые библиотеки
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

# Шаг 1: Загрузим данные
df = pd.read_csv('C:/Users/User/PycharmProjects/Lab6PRiS/dataset/dataset.csv')

# Настроим pandas на отображение всех столбцов
pd.set_option('display.max_columns', None)
# Посмотрим на все доступные столбцы
print("Все столбцы в датасете:")
print(df.columns)

# Шаг 2: Очистка данных
df_clean = df[['track_id', 'track_name', 'artists', 'popularity']]

# Проверим, что выбрали правильные столбцы
print("\nПример очищенных данных:")
print(df_clean.head())

# Шаг 3: Ограничиваем данные для матрицы предпочтений
# Например, ограничим количество строк для создания матрицы
df_small = df_clean.sample(n=5000, random_state=42)  # Выбираем 5000 случайных строк

# Создаем сводную таблицу на ограниченном наборе данных
user_track_matrix = pd.pivot_table(df_small, index='track_id', columns='track_name', values='popularity')

# Заполняем пропущенные значения нулями
user_track_matrix = user_track_matrix.fillna(0)

# Преобразуем данные в разреженную матрицу
user_track_matrix_sparse = csr_matrix(user_track_matrix.values)

# Шаг 4: Построение рекомендательной системы
model = NearestNeighbors(n_neighbors=5, algorithm='auto', metric='cosine')
model.fit(user_track_matrix_sparse)

# Шаг 5: Получаем рекомендации
# Получаем рекомендации для первого трека
distances, indices = model.kneighbors(user_track_matrix_sparse[0].reshape(1, -1))

# Получаем рекомендованные треки
recommended_tracks = user_track_matrix.columns[indices.flatten()]

# Шаг 6: Вывод рекомендаций
print("\nРекомендованные треки для первого трека:")
for track in recommended_tracks:
    print(track)

# Шаг 7: Визуализация (опционально)
recommended_data = df_clean[df_clean['track_name'].isin(recommended_tracks)]

plt.figure(figsize=(10, 6))
plt.bar(recommended_data['track_name'], recommended_data['popularity'])
plt.title("Популярность рекомендованных треков")
plt.xlabel("Треки")
plt.ylabel("Популярность")
plt.xticks(rotation=90)
plt.show()
