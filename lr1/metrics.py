"""Метрики задержки и пропускной способности. Чистые функции без обращения к серверу."""
import numpy as np


def ttft(t0, t1):
    """Время до первого токена: t1 - t0."""
    return t1 - t0


def tpot(t1, t_last, n_tokens):
    """Среднее время одного следующего токена: (t_N - t_1) / (N - 1).

    В знаменателе N - 1: между первым и последним токеном N - 1 интервалов.
    """
    if n_tokens is None or n_tokens < 2:
        return None
    return (t_last - t1) / (n_tokens - 1)


def e2e(t0, t_last):
    """Полное время ответа: t_N - t_0."""
    return t_last - t0


def percentile(values, q):
    """Процентиль с линейной интерполяцией (numpy по умолчанию)."""
    if len(values) == 0:
        raise ValueError("percentile: пустой список")
    return float(np.percentile(np.asarray(values, dtype=float), q))


def median_range(values):
    """Медиана и диапазон — для малого числа замеров (Правила, п. 7)."""
    if len(values) == 0:
        raise ValueError("median_range: пустой список")
    a = np.asarray(values, dtype=float)
    return {"n": int(len(a)), "median": float(np.median(a)), "min": float(a.min()), "max": float(a.max())}


def fit_line(x, y):
    """Прямая y = slope * x + intercept по методу наименьших квадратов и R^2."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 2:
        raise ValueError("fit_line: нужно не меньше двух точек")
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"slope": float(slope), "intercept": float(intercept),
            "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0}


def kv_bytes_per_token(n_layers, n_kv_heads, head_dim, bytes_per_value=2):
    """Размер KV-кеша на один токен: 2 * слои * головы KV * размерность головы * байт."""
    return 2 * n_layers * n_kv_heads * head_dim * bytes_per_value


def level_summary(records, wall_s, ttft_limit_s, tpot_limit_s):
    """Сводка уровня нагрузки из записей results.jsonl (прогрев не входит).

    Знаменатель — все запланированные запросы уровня: ошибочные запросы считаются
    неуспешными и в goodput не попадают, но остаются в числе запросов.
    """
    ok = [r for r in records if r["status"] == "ok"]
    n_total = len(records)
    out_tokens = sum(r["completion_tokens"] for r in ok)
    ttfts = [r["ttft_s"] for r in ok]
    tpots = [r["tpot_s"] for r in ok if r["tpot_s"] is not None]
    good = [r for r in ok if r["ttft_s"] <= ttft_limit_s
            and r["tpot_s"] is not None and r["tpot_s"] <= tpot_limit_s]
    summary = {
        "n_requests": n_total, "n_ok": len(ok), "n_error": n_total - len(ok),
        "wall_s": wall_s,
        "throughput_tok_s": out_tokens / wall_s if wall_s > 0 else None,
        "requests_per_s": len(ok) / wall_s if wall_s > 0 else None,
        "goodput_req_s": len(good) / wall_s if wall_s > 0 else None,
        "n_good": len(good),
    }
    if ttfts:
        summary.update(ttft_median_s=percentile(ttfts, 50), ttft_p90_s=percentile(ttfts, 90),
                       ttft_max_s=max(ttfts))
    if tpots:
        summary.update(tpot_median_s=percentile(tpots, 50), tpot_p90_s=percentile(tpots, 90))
    summary["within_limits"] = bool(
        ttfts and tpots and summary["n_error"] == 0
        and summary["ttft_p90_s"] <= ttft_limit_s and summary["tpot_median_s"] <= tpot_limit_s)
    return summary
