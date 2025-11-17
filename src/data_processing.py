import pandas as pd
import numpy as np

def prepare_funnel_data(df_raw):
    """Преобразование сырых данных в формат воронки"""
    df_pivot = df_raw.pivot_table(
        index=['dt', 'client_id'],
        columns='event',
        values='event',
        aggfunc='count',
        fill_value=0
    ).reset_index()

    df_pivot.columns = ['dt', 'client_id', 'add', 'click', 'view']

    # Бинарные флаги
    df_funnel = df_pivot.copy()
    for col in ['view', 'click', 'add']:
        df_funnel[col] = (df_funnel[col] > 0).astype(int)

    return df_funnel

def calculate_user_metrics(df_funnel):
    """Расчет метрик на уровне пользователя"""
    df_user = df_funnel.groupby('client_id').agg({
        'view': 'sum',
        'click': 'sum',
        'add': 'sum'
    }).reset_index()

    df_user['CTR'] = df_user.apply(lambda row: row['click'] / row['view'] if row['view'] > 0 else 0, axis=1)
    df_user['CR'] = df_user.apply(lambda row: row['add'] / row['click'] if row['click'] > 0 else 0, axis=1)

    return df_user
