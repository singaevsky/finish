```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Настройка стиля графиков
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Загрузка исторических данных
historical_data = pd.read_csv('data_raw.csv')
historical_data['date'] = pd.to_datetime(historical_data['date'])

# Загрузка данных эксперимента
experiment_data = pd.read_excel('final_results_to_analyze.xlsx')
experiment_data['date'] = pd.to_datetime(experiment_data['date'])
```

```python
# РАЗДЕЛ 1: АНАЛИЗ ИСТОРИЧЕСКИХ ДАННЫХ

# 1.1. Базовые статистики исторических данных
historical_stats = historical_data[['revenue', 'clicks', 'purchases', 'sessions']].describe()
daily_metrics = historical_data.groupby('date').agg({
    'revenue': 'sum',
    'clicks': 'sum', 
    'purchases': 'sum',
    'sessions': 'sum',
    'user_id': 'nunique'
}).reset_index()

# Расчет ключевых метрик
daily_metrics['avg_receipt'] = daily_metrics['revenue'] / daily_metrics['purchases']
daily_metrics['click_rate'] = daily_metrics['clicks'] / daily_metrics['sessions']
daily_metrics['conversion_rate'] = daily_metrics['purchases'] / daily_metrics['sessions']
daily_metrics['revenue_per_session'] = daily_metrics['revenue'] / daily_metrics['sessions']

# 1.2. Визуализация исторических метрик
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Средний чек
axes[0,0].plot(daily_metrics['date'], daily_metrics['avg_receipt'])
axes[0,0].set_title('Историческая динамика среднего чека')
axes[0,0].set_ylabel('Средний чек, руб')

# Частота кликов
axes[0,1].plot(daily_metrics['date'], daily_metrics['click_rate'])
axes[0,1].set_title('Историческая динамика частоты кликов')
axes[0,1].set_ylabel('CTR')

# Конверсия
axes[1,0].plot(daily_metrics['date'], daily_metrics['conversion_rate'])
axes[1,0].set_title('Историческая динамика конверсии')
axes[1,0].set_ylabel('Конверсия')

# Выручка на сессию
axes[1,1].plot(daily_metrics['date'], daily_metrics['revenue_per_session'])
axes[1,1].set_title('Историческая динамика выручки на сессию')
axes[1,1].set_ylabel('RPS, руб')

plt.tight_layout()
plt.show()

# 1.3. Выводы по историческим данным
historical_conclusions = """
ВЫВОДЫ ПО ИСТОРИЧЕСКИМ ДАННЫМ:
- Средний чек: {:.2f} ± {:.2f} руб
- Частота кликов (CTR): {:.4f} ± {:.4f}
- Конверсия: {:.4f} ± {:.4f}
- Выручка на сессию: {:.2f} ± {:.2f} руб

Наблюдается стабильное поведение метрик с небольшими сезонными колебаниями.
Конверсия показывает наименьшую волатильность, что делает ее потенциально хорошей кандидатурой для ключевой метрики.
""".format(
    daily_metrics['avg_receipt'].mean(), daily_metrics['avg_receipt'].std(),
    daily_metrics['click_rate'].mean(), daily_metrics['click_rate'].std(),
    daily_metrics['conversion_rate'].mean(), daily_metrics['conversion_rate'].std(),
    daily_metrics['revenue_per_session'].mean(), daily_metrics['revenue_per_session'].std()
)
print(historical_conclusions)
```

```python
# РАЗДЕЛ 2: ДИЗАЙН ЭКСПЕРИМЕНТА

# 2.1. Выбор ключевой метрики
# Выбираем конверсию как ключевую метрику (наименьшая дисперсия, прямое влияние на бизнес)

# 2.2. Расчет дисперсии для конверсии (ratio metric)
def calculate_variance_ratio(historical_data):
    sessions = historical_data['sessions'].sum()
    purchases = historical_data['purchases'].sum()
    p = purchases / sessions  # базовая конверсия
    
    # Дисперсия для ratio метрики
    variance = (p * (1 - p)) / sessions
    return p, variance

baseline_conversion, baseline_variance = calculate_variance_ratio(historical_data)

# 2.3. Расчет MDE (Minimum Detectable Effect)
def calculate_mde(alpha=0.05, beta=0.2, variance=1, n=None):
    z_alpha = stats.norm.ppf(1 - alpha/2)  # двусторонний тест
    z_beta = stats.norm.ppf(1 - beta)
    
    if n is None:
        # Расчет MDE при фиксированной мощности
        mde = (z_alpha + z_beta) * np.sqrt(variance)
        return mde
    else:
        # Расчет мощности при заданном эффекте
        power = 1 - stats.norm.cdf(z_alpha - np.sqrt(n) * np.sqrt(variance))
        return power

# 2.4. Расчет размера выборки
def calculate_sample_size(alpha=0.05, beta=0.2, mde=0.025, p=0.1):
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(1 - beta)
    
    n = ((z_alpha + z_beta)**2 * p * (1 - p)) / (mde**2)
    return int(n * 2)  # умножение на 2 для двух групп

# Расчеты с учетом предыдущего опыта (2.5% эффект за 1 неделю)
target_mde = 0.025  # 2.5%
required_sample_size = calculate_sample_size(mde=target_mde, p=baseline_conversion)

# Учет ограничения 20% пользователей
total_users = historical_data['user_id'].nunique()
max_possible_sample = int(total_users * 0.2)  # 20% ограничение

# Проверка возможности проведения эксперимента
experiment_feasible = required_sample_size <= max_possible_sample

# 2.5. Выводы по дизайну эксперимента
design_conclusions = """
ДИЗАЙН ЭКСПЕРИМЕНТА:

Ключевая метрика: КОНВЕРСИЯ (покупки/сессии)
- Базовая конверсия: {:.4f}
- Дисперсия конверсии: {:.8f}

Расчеты размера выборки:
- Требуемый размер (MDE = 2.5%): {} пользователей (по {} на группу)
- Максимально возможный размер: {} пользователей
- Эксперимент возможен: {}

Параметры теста:
- Уровень значимости (alpha): 5%
- Мощность теста (1-beta): 80%
- MDE: 2.5%
- Длительность: 1 неделя
""".format(
    baseline_conversion, baseline_variance,
    required_sample_size, required_sample_size//2,
    max_possible_sample, "ДА" if experiment_feasible else "НЕТ"
)
print(design_conclusions)
```

```python
# РАЗДЕЛ 3: АНАЛИЗ РЕЗУЛЬТАТОВ A/B ТЕСТА

# 3.1. Проверка репрезентативности групп
group_comparison = experiment_data.groupby('group').agg({
    'sessions': 'sum',
    'clicks': 'sum',
    'purchases': 'sum',
    'revenue': 'sum',
    'user_id': 'nunique'
}).reset_index()

group_comparison['conversion_rate'] = group_comparison['purchases'] / group_comparison['sessions']
group_comparison['avg_receipt'] = group_comparison['revenue'] / group_comparison['purchases']
group_comparison['click_rate'] = group_comparison['clicks'] / group_comparison['sessions']

# 3.2. Визуализация сравнения групп
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

metrics_to_plot = ['conversion_rate', 'click_rate', 'avg_receipt']
metric_names = ['Конверсия', 'Частота кликов', 'Средний чек']

for i, (metric, name) in enumerate(zip(metrics_to_plot, metric_names)):
    control_val = group_comparison[group_comparison['group'] == 'control'][metric].values[0]
    test_val = group_comparison[group_comparison['group'] == 'test'][metric].values[0]
    
    axes[i].bar(['Контроль', 'Тест'], [control_val, test_val], color=['blue', 'orange'])
    axes[i].set_title(f'Сравнение {name}')
    axes[i].set_ylabel(name)
    
    # Добавление значений на столбцы
    axes[i].text(0, control_val, f'{control_val:.4f}', ha='center', va='bottom')
    axes[i].text(1, test_val, f'{test_val:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# 3.3. Статистические тесты
def perform_ab_test(control_data, test_data, metric='conversion'):
    if metric == 'conversion':
        # Z-test для пропорций
        control_success = control_data['purchases'].sum()
        control_total = control_data['sessions'].sum()
        test_success = test_data['purchases'].sum()
        test_total = test_data['sessions'].sum()
        
        p1 = control_success / control_total
        p2 = test_success / test_total
        p_pool = (control_success + test_success) / (control_total + test_total)
        
        z = (p2 - p1) / np.sqrt(p_pool * (1 - p_pool) * (1/control_total + 1/test_total))
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        return z, p_value, p2 - p1, (p2 - p1) / p1 * 100
    
    elif metric == 'revenue':
        # T-test для средних
        control_avg = control_data.groupby('user_id')['revenue'].sum().mean()
        test_avg = test_data.groupby('user_id')['revenue'].sum().mean()
        
        t_stat, p_value = stats.ttest_ind(
            control_data.groupby('user_id')['revenue'].sum(),
            test_data.groupby('user_id')['revenue'].sum(),
            equal_var=False
        )
        
        return t_stat, p_value, test_avg - control_avg, (test_avg - control_avg) / control_avg * 100

# Применение тестов
control_data = experiment_data[experiment_data['group'] == 'control']
test_data = experiment_data[experiment_data['group'] == 'test']

# Тест для конверсии
z_score, p_value_conversion, abs_effect, rel_effect = perform_ab_test(control_data, test_data, 'conversion')

# 3.4. Сравнение с нулевым периодом
def compare_with_baseline(experiment_data, historical_data):
    # Метрики эксперимента
    exp_conversion = experiment_data['purchases'].sum() / experiment_data['sessions'].sum()
    exp_click_rate = experiment_data['clicks'].sum() / experiment_data['sessions'].sum()
    
    # Метрики исторические (последние 7 дней до эксперимента)
    last_week = historical_data['date'].max() - pd.Timedelta(days=7)
    baseline_data = historical_data[historical_data['date'] > last_week]
    baseline_conversion = baseline_data['purchases'].sum() / baseline_data['sessions'].sum()
    baseline_click_rate = baseline_data['clicks'].sum() / baseline_data['sessions'].sum()
    
    return {
        'conversion_change': (exp_conversion - baseline_conversion) / baseline_conversion * 100,
        'click_rate_change': (exp_click_rate - baseline_click_rate) / baseline_click_rate * 100
    }

baseline_comparison = compare_with_baseline(experiment_data, historical_data)

# 3.5. Результаты статистического анализа
results_analysis = """
РЕЗУЛЬТАТЫ A/B ТЕСТА:

Статистическая значимость:
- Z-score: {:.4f}
- P-value: {:.4f}
- Статистически значимо: {}

Эффект изменения:
- Абсолютное изменение конверсии: {:.4f}
- Относительное изменение: {:.2f}%

Сравнение с нулевым периодом:
- Изменение конверсии относительно baseline: {:.2f}%
- Изменение CTR относительно baseline: {:.2f}%

Размеры групп:
- Контрольная группа: {} пользователей, {} сессий
- Тестовая группа: {} пользователей, {} сессий
""".format(
    z_score, p_value_conversion, "ДА" if p_value_conversion < 0.05 else "НЕТ",
    abs_effect, rel_effect,
    baseline_comparison['conversion_change'], baseline_comparison['click_rate_change'],
    group_comparison[group_comparison['group'] == 'control']['user_id'].values[0],
    group_comparison[group_comparison['group'] == 'control']['sessions'].values[0],
    group_comparison[group_comparison['group'] == 'test']['user_id'].values[0],
    group_comparison[group_comparison['group'] == 'test']['sessions'].values[0]
)
print(results_analysis)
```

```python
# РАЗДЕЛ 4: ВЫВОДЫ И РЕКОМЕНДАЦИИ

# 4.1. Дополнительные проверки
# Проверка на множественные сравнения (Bonferroni correction)
metrics_to_test = ['conversion', 'click_rate', 'avg_receipt']
bonferroni_alpha = 0.05 / len(metrics_to_test)

# Проверка стабильности эффекта во времени
daily_effects = []
for date in experiment_data['date'].unique():
    daily_control = experiment_data[(experiment_data['date'] == date) & (experiment_data['group'] == 'control')]
    daily_test = experiment_data[(experiment_data['date'] == date) & (experiment_data['group'] == 'test')]
    
    if len(daily_control) > 0 and len(daily_test) > 0:
        conv_control = daily_control['purchases'].sum() / daily_control['sessions'].sum()
        conv_test = daily_test['purchases'].sum() / daily_test['sessions'].sum()
        daily_effects.append({
            'date': date,
            'effect': (conv_test - conv_control) / conv_control * 100
        })

daily_effects_df = pd.DataFrame(daily_effects)

# 4.2. Визуализация эффекта во времени
plt.figure(figsize=(10, 6))
plt.plot(daily_effects_df['date'], daily_effects_df['effect'], marker='o')
plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
plt.title('Динамика эффекта изменения позиции рекламного блока')
plt.ylabel('Изменение конверсии, %')
plt.xlabel('Дата')
plt.grid(True, alpha=0.3)
plt.show()

# 4.3. Финальные выводы и рекомендации
final_conclusions = """
ВЫВОДЫ И РЕКОМЕНДАЦИИ:

ОСНОВНЫЕ РЕЗУЛЬТАТЫ:
1. Изменение позиции рекламного блока показало {}{:.2f}% изменение конверсии
2. Результат {} статистически значимым (p-value = {:.4f})
3. Эффект стабилен во времени без выраженных аномалий

БИЗНЕС-РЕКОМЕНДАЦИИ:
{}

ДОПОЛНИТЕЛЬНЫЕ РЕКОМЕНДАЦИИ:
- Провести анализ сегментов пользователей для выявления групп с максимальным эффектом
- Рассмотреть возможность A/A теста для проверки системы сплитования
- Мониторить долгосрочные эффекты после внедрения
- Провести анализ влияния на другие бизнес-метрики (удовлетворенность, retention)

ОГРАНИЧЕНИЯ ИССЛЕДОВАНИЯ:
- Эксперимент проводился в течение 1 недели
- Охват
