"""Загрузка и проверка конфигурации запуска."""
import copy
from pathlib import Path

import yaml


class ConfigError(ValueError):
    """Конфигурация некорректна: запуск не начинается."""


KINDS = ("prompt_length", "prefix_cache", "output_length", "load", "checks")

DEFAULTS = {
    "experiment": "lr1",
    "series": "debug",
    "kind": None,
    "endpoint_file": "endpoint.yaml",
    "book": {"path": "data/pg1342.txt"},
    "request": {
        "instruction": "Summarize the following text in one sentence.",
        "ignore_eos": True,
        "cache_prompt": False,
    },
}


def deep_merge(base, override):
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _pos_int(errors, name, value):
    if not _is_int(value) or value < 1:
        errors.append(f"{name} должен быть целым >= 1, получено {value!r}")


def _pos_int_list(errors, name, value):
    if not isinstance(value, list) or not value or not all(_is_int(v) and v >= 1 for v in value):
        errors.append(f"{name} должен быть непустым списком целых >= 1, получено {value!r}")


def validate(cfg):
    """Проверяет конфигурацию до начала запуска; при ошибке бросает ConfigError."""
    errors = []
    kind = cfg.get("kind")
    if kind not in KINDS:
        errors.append(f"kind должен быть одним из {KINDS}, получено {kind!r}")

    ep = cfg.get("endpoint", {})
    for key in ("base_url", "server_root", "model"):
        if not isinstance(ep.get(key), str) or not ep[key]:
            errors.append(f"endpoint.{key} должен быть непустой строкой, получено {ep.get(key)!r}")
    if not _is_num(ep.get("timeout_s")) or ep["timeout_s"] <= 0:
        errors.append(f"endpoint.timeout_s должен быть > 0, получено {ep.get('timeout_s')!r}")

    gen = cfg.get("generation", {})
    if not _is_num(gen.get("temperature")) or gen["temperature"] < 0:
        errors.append(f"generation.temperature должна быть задана явно и >= 0, получено {gen.get('temperature')!r}")

    if kind == "prompt_length":
        _pos_int_list(errors, "lengths", cfg.get("lengths"))
        _pos_int(errors, "repeats", cfg.get("repeats"))
        _pos_int(errors, "max_tokens", cfg.get("max_tokens"))
    elif kind == "prefix_cache":
        _pos_int(errors, "prompt_tokens", cfg.get("prompt_tokens"))
        _pos_int(errors, "times", cfg.get("times"))
        _pos_int(errors, "max_tokens", cfg.get("max_tokens"))
    elif kind == "output_length":
        _pos_int(errors, "prompt_tokens", cfg.get("prompt_tokens"))
        _pos_int_list(errors, "output_lengths", cfg.get("output_lengths"))
        _pos_int(errors, "repeats", cfg.get("repeats"))
    elif kind == "load":
        _pos_int(errors, "prompt_tokens", cfg.get("prompt_tokens"))
        _pos_int(errors, "max_tokens", cfg.get("max_tokens"))
        _pos_int_list(errors, "levels", cfg.get("levels"))
        _pos_int(errors, "min_requests", cfg.get("min_requests"))
        _pos_int(errors, "requests_per_client", cfg.get("requests_per_client"))
        lim = cfg.get("limits", {})
        for key in ("ttft_p90_s", "tpot_median_s"):
            if not _is_num(lim.get(key)) or lim[key] <= 0:
                errors.append(f"limits.{key} должен быть > 0, получено {lim.get(key)!r}")

    if errors:
        raise ConfigError("Некорректная конфигурация:\n  - " + "\n  - ".join(errors))
    return cfg


def load_config(path, overrides=None, root=None):
    """Читает YAML, подмешивает endpoint.yaml и значения по умолчанию, проверяет."""
    root = Path(root) if root else Path.cwd()
    with open(Path(path), encoding="utf-8") as f:
        from_file = yaml.safe_load(f) or {}
    cfg = deep_merge(DEFAULTS, from_file)
    cfg = deep_merge(cfg, overrides or {})
    with open(root / cfg["endpoint_file"], encoding="utf-8") as f:
        endpoint = yaml.safe_load(f) or {}
    cfg["endpoint"] = deep_merge(endpoint.get("endpoint", {}), cfg.get("endpoint", {}))
    cfg["generation"] = deep_merge(endpoint.get("generation", {}), cfg.get("generation", {}))
    return validate(cfg)
