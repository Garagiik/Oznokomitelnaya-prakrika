import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging

#  1. ЗАГРУЗКА ДАННЫХ 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Загрузка CSV-файла")
# Пропускаем первые 3 строки с метаданными
df = pd.read_csv('Data.csv', skiprows=3, delimiter=',')
logger.info(f"Загружено строк: {len(df)}")

#  2. ОЧИСТКА ДАННЫХ
logger.info("Очистка данных")
df = df.rename(columns={'temperature_2m (°C)': 'temperature'})
temp = df['temperature'].copy()
temp = temp.astype(float)

# Удаляем явные выбросы 
mean, std = temp.mean(), temp.std()
lower, upper = mean - 3*std, mean + 3*std
outliers = (temp < lower) | (temp > upper)
logger.info(f"Найдено выбросов: {outliers.sum()}")

# Очищаем: выбросы -> NaN
temp_clean = temp.copy()
temp_clean[outliers] = np.nan
logger.info(f"После удаления выбросов (замена на NaN) осталось пропусков: {temp_clean.isna().sum()}")

# 3. МАССИВ ДЛЯ РАБОТЫ
data_array = temp_clean.dropna().to_numpy()
logger.info(f"Сформирован массив NumPy размером: {len(data_array)} (без пропусков)")

# 6. АНАЛИЗ ПРОПУСКОВ И ЗАПОЛНЕНИЕ
logger.info("Анализ пропусков во временном ряде")
null_count = temp_clean.isna().sum()
logger.info(f"Пропусков до заполнения: {null_count} (включая удалённые выбросы)")

# Заполняем медианой
median_val = temp_clean.median()
temp_filled = temp_clean.fillna(median_val)
logger.info(f"Пропуски заполнены медианой = {median_val:.2f}")
logger.info(f"Пропусков после заполнения: {temp_filled.isna().sum()}")

# 4. РАСЧЁТ ХАРАКТЕРИСТИК  
logger.info("Расчёт статистических характеристик (дисперсия, σ, размах, CV, Q3)")

variance = np.var(data_array, ddof=1)           # 11. Дисперсия
std_dev = np.std(data_array, ddof=1)            # 12. Среднеквадратическое
data_range = np.max(data_array) - np.min(data_array)  # 13. Размах
cv = (std_dev / np.mean(data_array)) * 100      # 14. Коэффициент вариации (%)
q3 = np.percentile(data_array, 75)              # 15. Третий квартиль

print("\n" + "="*50)
print("СТАТИСТИЧЕСКИЕ ХАРАКТЕРИСТИКИ (вариант 26)")
print("="*50)
print(f"Дисперсия                     : {variance:.4f}")
print(f"Среднеквадратическое отклонение: {std_dev:.4f}")
print(f"Размах                         : {data_range:.4f}")
print(f"Коэффициент вариации (%)       : {cv:.2f}%")
print(f"Третий квартиль (Q3)           : {q3:.4f}")
print("="*50 + "\n")

# 8. ТРОЙКА НАИБОЛЬШИХ И НАИМЕНЬШИХ ЗНАЧЕНИЙ 
logger.info("Поиск трёх наибольших и наименьших значений")
sorted_vals = np.sort(temp_filled)
smallest_3 = sorted_vals[:3]
largest_3 = sorted_vals[-3:]

print("ТРИ НАИМЕНЬШИХ ЗНАЧЕНИЯ:", smallest_3)
print("ТРИ НАИБОЛЬШИХ ЗНАЧЕНИЯ:", largest_3)
print()

#  7. ГРУППИРОВКА ПО ДНЯМ НЕДЕЛИ
logger.info("Группировка по дням недели")
df['time'] = pd.to_datetime(df['time'])
df['day_of_week'] = df['time'].dt.day_name()
df['temp_filled'] = temp_filled

grouped = df.groupby('day_of_week')['temp_filled'].agg(['mean', 'median', 'std', 'count'])
# Правильный порядок дней
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
grouped = grouped.reindex(days_order)

print("\n" + "="*60)
print("ГРУППИРОВКА ПО ДНЯМ НЕДЕЛИ (средняя, медиана, σ, количество)")
print("="*60)
print(grouped.round(2))
print("="*60 + "\n")

# 5. ВИЗУАЛИЗАЦИЯ 2,3,4
logger.info("Построение графиков: плотность, гистограмма, ящик с усами")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 2. График плотности распределения
sns.kdeplot(data_array, ax=axes[0], fill=True, color='skyblue')
axes[0].set_title('Плотность распределения температуры')
axes[0].set_xlabel('Температура (°C)')

# 3. Гистограмма
axes[1].hist(data_array, bins=30, edgecolor='black', alpha=0.7, color='lightgreen')
axes[1].set_title('Гистограмма температуры')
axes[1].set_xlabel('Температура (°C)')
axes[1].set_ylabel('Частота')

# 4. Ящик с усами (boxplot)
axes[2].boxplot(data_array, vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightblue'))
axes[2].set_title('Ящик с усами (Boxplot)')
axes[2].set_ylabel('Температура (°C)')
axes[2].set_xticklabels(['Температура'])

plt.tight_layout()
plt.show()

# 9. ПРОВЕРКА ГИПОТЕЗЫ 
logger.info("Проверка гипотезы о наличии тренда (корреляция времени и температуры)")
# Создаём числовой индекс времени
time_numeric = np.arange(len(temp_filled))
# Убираем NaN на всякий случай
mask = ~np.isnan(temp_filled)
time_clean = time_numeric[mask]
temp_clean2 = temp_filled[mask]

# Корреляция Пирсона
corr, p_value = stats.pearsonr(time_clean, temp_clean2)

print("="*50)
print("ПРОВЕРКА ГИПОТЕЗЫ: существует ли линейный тренд?")
print("="*50)
print(f"Коэффициент корреляции Пирсона (время → температура): {corr:.4f}")
print(f"p-значение: {p_value:.6f}")

alpha = 0.05
if p_value < alpha:
    print("→ Отвергаем H₀: тренд статистически значим (p < 0.05).")
    print("  В данных присутствует значимый линейный тренд (температура растёт со временем).")
else:
    print("→ Не отвергаем H₀: значимый тренд не обнаружен.")

logger.info("Анализ полностью завершён.")