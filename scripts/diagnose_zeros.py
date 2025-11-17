import os
import sys
import pandas as pd
import numpy as np

# Найти папку Data (поднимаемся вверх до 6 уровней)
current_dir = os.path.abspath(os.getcwd())
BASE_DIR = current_dir
found = False
for _ in range(6):
    candidate = os.path.join(BASE_DIR, 'Data')
    if os.path.isdir(candidate):
        found = True
        break
    parent = os.path.dirname(BASE_DIR)
    if parent == BASE_DIR:
        break
    BASE_DIR = parent

if not found:
    print('⚠️ Папка Data не найдена в пределах 6 уровней от рабочей директории:', os.getcwd())

DATA_RAW = os.path.join(BASE_DIR, 'Data', 'data_raw.csv')
FINAL_XLSX = os.path.join(BASE_DIR, 'Data', 'final_results_to_analyze.xlsx')

print('BASE_DIR =', BASE_DIR)
print('DATA_RAW =', DATA_RAW, 'exists=', os.path.exists(DATA_RAW))
print('FINAL_XLSX =', FINAL_XLSX, 'exists=', os.path.exists(FINAL_XLSX))

# Функция безопасного чтения CSV (utf-8 -> cp1251)
def read_csv_safe(path):
    for enc in ('utf-8', 'cp1251', 'latin1'):
        try:
            df = pd.read_csv(path, parse_dates=[col for col in ['dt', 'date'] if col in pd.read_csv(path, nrows=0).columns], encoding=enc)
            print(f'Read CSV with encoding={enc}')
            return df
        except Exception as e:
            # попробуем следующую кодировку
            last_err = e
    print('ERROR reading CSV:', last_err)
    raise last_err

# Чтение data_raw
if os.path.exists(DATA_RAW):
    try:
        df_raw = read_csv_safe(DATA_RAW)
    except Exception as e:
        print('Не удалось прочитать data_raw.csv:', e)
        sys.exit(1)
    print('\n== data_raw overview ==')
    print('shape:', df_raw.shape)
    print('columns:', list(df_raw.columns))
    if 'event_type' in df_raw.columns:
        print('\nUnique event_type (top 30):')
        print(df_raw['event_type'].value_counts().head(30))
    else:
        print("Колонка 'event_type' не найдена в data_raw.csv")

    # Попытка собрать воронку
    if set(['dt','client_id','event_type']).issubset(df_raw.columns):
        funnel = df_raw.groupby(['dt', 'client_id', 'event_type']).size().unstack(fill_value=0)
        print('\nFunnel columns (sample up to 50):', funnel.columns.tolist()[:50])
        # Проверим ожидаемые имена
        expected = ['view','click','add','views','clicks','adds']
        for e in expected:
            if e in funnel.columns:
                s = funnel[e].sum()
                print(f"Sum of funnel column '{e}': {s}")
            else:
                print(f"Column '{e}' NOT present in funnel columns")
        # Сколько пользователей имеют хоть одно событие
        any_event = (funnel.sum(axis=1) > 0).sum()
        total_rows = funnel.shape[0]
        print(f"Rows with any event >0: {any_event} / {total_rows}")
    else:
        print("Невозможно построить funnel — нет одной из колонок: dt, client_id, event_type")
else:
    print('\nФайл data_raw.csv отсутствует — пропускаем диагностику data_raw')

# Чтение final_results_to_analyze.xlsx
if os.path.exists(FINAL_XLSX):
    try:
        df_res = pd.read_excel(FINAL_XLSX)
    except Exception as e:
        print('Не удалось прочитать final_results_to_analyze.xlsx:', e)
        sys.exit(1)
    print('\n== final_results_to_analyze overview ==')
    print('shape:', df_res.shape)
    print('columns:', list(df_res.columns))
    print('\nColumn value counts (showing ones/zeros or top values):')
    for col in ['is_view_ads', 'is_adds_ads', 'cnt_view_ads', 'cnt_adds_ads', 'cnt_orders_ads', 'sum_adds_ads', 'sum_orders_ads', 'ab_group']:
        if col in df_res.columns:
            try:
                vc = df_res[col].value_counts(dropna=False)
                print(f"{col}:\n{vc.head(10)}")
                if np.issubdtype(df_res[col].dtype, np.number):
                    print(f"  sum={df_res[col].sum()}  mean={df_res[col].mean():.6f}")
            except Exception as e:
                print(f"  cannot value_counts for {col}: {e}")
        else:
            print(f"{col}: NOT FOUND")

    # Проверим агрегацию по пользователю
    if 'client_id' in df_res.columns:
        user_ab = df_res.groupby('client_id').agg(
            ab_group=('ab_group','first') if 'ab_group' in df_res.columns else ('client_id','first'),
            has_click=('is_view_ads','max') if 'is_view_ads' in df_res.columns else ('client_id','count'),
            has_add=('is_adds_ads','max') if 'is_adds_ads' in df_res.columns else ('client_id','count'),
            adds_count=('cnt_adds_ads','sum') if 'cnt_adds_ads' in df_res.columns else ('client_id','count')
        )
        print('\nUser-level sample (first 10 rows):')
        print(user_ab.head(10))
    else:
        print('NO client_id in final_results_to_analyze.xlsx')
else:
    print('\nФайл final_results_to_analyze.xlsx отсутствует — пропускаем диагностику final_results')

print('\n✅ Диагностика завершена')
