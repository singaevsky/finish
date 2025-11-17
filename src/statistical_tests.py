from scipy import stats
import numpy as np

def z_test_proportions(success_a, n_a, success_b, n_b):
    """Z-тест для разницы пропорций"""
    p1 = success_a / n_a
    p2 = success_b / n_b
    p_pool = (success_a + success_b) / (n_a + n_b)

    z = (p1 - p2) / np.sqrt(p_pool * (1-p_pool) * (1/n_a + 1/n_b))
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return z, p_value

def confidence_interval(success, n, confidence=0.95):
    """Доверительный интервал для пропорции"""
    p = success / n
    z = stats.norm.ppf(1 - (1-confidence)/2)
    margin = z * np.sqrt(p * (1-p) / n)
    return p, p - margin, p + margin

def calculate_mde(mu, std, sample_size, alpha=0.05, beta=0.2):
    """Расчет минимального детектируемого эффекта"""
    t_alpha = stats.norm.ppf(1 - alpha/2)
    t_beta = stats.norm.ppf(1 - beta)
    correction = 2 + 1  # для двух групп
    mde_abs = np.sqrt(correction) * (t_alpha + t_beta) * std / np.sqrt(sample_size)
    mde_rel = mde_abs * 100 / mu
    return mde_abs, mde_rel
