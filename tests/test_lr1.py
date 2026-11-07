"""Проверки реализации ЛР1 (п. 3.4 задания и устройство клиента). Результаты в таблицы отчёта не входят.

    python -m pytest -q

Сервер модели не нужен: используется заглушка tests/fake_server.py. Проверки на настоящем
сервере (точная длина ответа, повтор при температуре 0, KV-кеш) — серия `checks`.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_server import FakeLLM  # noqa: E402

from lr1 import metrics  # noqa: E402
from lr1.client import StreamTimer, run_request  # noqa: E402
from lr1.config import ConfigError, load_config, validate  # noqa: E402
from lr1.experiments import Ctx, summarize  # noqa: E402
from lr1.prompts import PromptBuilder, ServerTokenizer, load_book  # noqa: E402

COPY_ITEMS = ["lr1", "configs", "data", "endpoint.yaml", "run.py", "prepare_data.py", ".gitignore", ".gitattributes"]


# ---------- арифметика метрик ----------

def test_ttft_tpot_arithmetic():
    """t0 = 0, t1 = 0,4 с, tN = 2,4 с, N = 101 дают TTFT 400 мс и TPOT 20 мс."""
    assert metrics.ttft(0, 0.4) == pytest.approx(0.4)
    assert metrics.tpot(0.4, 2.4, 101) == pytest.approx(0.020)
    assert metrics.e2e(0, 2.4) == pytest.approx(2.4)


def test_tpot_single_token_is_none():
    assert metrics.tpot(1.0, 1.0, 1) is None


def test_percentile_and_median_range():
    assert metrics.percentile([1, 2, 3, 4, 5], 50) == 3
    assert metrics.percentile([0, 10], 90) == pytest.approx(9.0)
    mr = metrics.median_range([3, 1, 2, 10, 4])
    assert (mr["median"], mr["min"], mr["max"], mr["n"]) == (3, 1, 10, 5)
    with pytest.raises(ValueError):
        metrics.percentile([], 50)


def test_fit_line_recovers_slope_and_intercept():
    x = [16, 64, 256, 1024]
    y = [0.5 + 0.1 * v for v in x]
    fit = metrics.fit_line(x, y)
    assert fit["slope"] == pytest.approx(0.1) and fit["intercept"] == pytest.approx(0.5) and fit["r2"] == pytest.approx(1)


def test_kv_bytes_per_token_matches_task_text():
    """У Qwen3-4B-Instruct-2507: 36 слоёв, 8 голов KV, размерность 128, FP16 → 147 456 байт = 144 КиБ."""
    assert metrics.kv_bytes_per_token(36, 8, 128, 2) == 147_456


def test_level_summary_counts_errors_in_denominator():
    ok = lambda ttft, tpot: {"status": "ok", "completion_tokens": 100, "ttft_s": ttft, "tpot_s": tpot}  # noqa: E731
    recs = [ok(1, 0.05), ok(2, 0.05), ok(3, 0.2), {"status": "error", "completion_tokens": None, "ttft_s": None, "tpot_s": None}]
    s = metrics.level_summary(recs, wall_s=10, ttft_limit_s=2.5, tpot_limit_s=0.1)
    assert s["n_requests"] == 4 and s["n_ok"] == 3 and s["n_error"] == 1
    assert s["throughput_tok_s"] == pytest.approx(30)
    assert s["n_good"] == 2 and s["goodput_req_s"] == pytest.approx(0.2)
    assert s["within_limits"] is False  # есть ошибочный запрос


# ---------- разбор потока ----------

def _chunk(content=None, usage=None, finish=None, choices=True, timings=None):
    delta = SimpleNamespace(content=content)
    ch = [SimpleNamespace(delta=delta, finish_reason=finish)] if choices else []
    return SimpleNamespace(choices=ch, usage=usage, model_extra={"timings": timings} if timings else {})


def test_stream_timer_uses_usage_not_chunk_count():
    t = StreamTimer(t0=0.0)
    t.feed(_chunk(None), 0.1)                 # фрагмент с ролью, текста нет
    t.feed(_chunk("a b"), 1.0)                # t1: первый фрагмент с текстом
    t.feed(_chunk("c"), 1.5)
    t.feed(_chunk("d e f", finish="length"), 2.0)
    t.feed(_chunk(choices=False, usage=SimpleNamespace(prompt_tokens=50, completion_tokens=10)), 2.1)
    r = t.result()
    assert r["status"] == "ok" and r["n_chunks"] == 3 and r["completion_tokens"] == 10
    assert r["ttft_s"] == pytest.approx(1.0)
    assert r["tpot_s"] == pytest.approx((2.0 - 1.0) / 9)   # N - 1 в знаменателе, N из usage
    assert r["e2e_s"] == pytest.approx(2.0)


def test_stream_timer_without_usage_is_error():
    t = StreamTimer(t0=0.0)
    t.feed(_chunk("x"), 1.0)
    assert t.result()["status"] == "error"


# ---------- конфигурация ----------

def test_config_requires_explicit_temperature():
    cfg = {"kind": "checks", "endpoint": {"base_url": "u", "server_root": "r", "model": "m", "timeout_s": 5},
           "generation": {}}
    with pytest.raises(ConfigError, match="temperature"):
        validate(cfg)


def test_config_rejects_bad_lengths(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("kind: prompt_length\nlengths: [128, -5]\nrepeats: 0\nmax_tokens: 16\n", encoding="utf-8")
    with pytest.raises(ConfigError) as e:
        load_config(bad, root=ROOT)
    assert "lengths" in str(e.value) and "repeats" in str(e.value)


def test_real_configs_are_valid():
    for path in sorted((ROOT / "configs").glob("*.yaml")):
        cfg = load_config(path, root=ROOT)
        assert cfg["generation"]["temperature"] == 0


# ---------- клиент и генератор запросов против заглушки ----------

@pytest.fixture()
def fake():
    with FakeLLM(prefill_s_per_token=0.0002, token_s=0.005) as f:
        yield f


def _cfg(fake_server, tmp_path=None, **extra):
    cfg = {"kind": "prompt_length", "series": "t", "request": {"ignore_eos": True, "cache_prompt": False},
           "endpoint": {"base_url": fake_server.root + "/v1", "server_root": fake_server.root, "model": "fake",
                        "api_key": "none", "timeout_s": 30},
           "generation": {"temperature": 0.0}, "book": {"path": str(ROOT / "data/pg1342.txt")}}
    cfg.update(extra)
    return cfg


def test_client_measures_and_checks_length(fake):
    from openai import OpenAI

    cfg = _cfg(fake)
    client = OpenAI(base_url=cfg["endpoint"]["base_url"], api_key="none", max_retries=0)
    req = {"id": "r1", "condition": {"c": 1}, "messages": [{"role": "user", "content": "x" * 100}], "max_tokens": 64}
    rec = run_request(client, cfg, req)
    assert rec["status"] == "ok" and rec["completion_tokens"] == 64          # ignore_eos: ровно 64 токена
    assert rec["prompt_tokens"] == 104
    assert rec["ttft_s"] >= 100 * 0.0002 and rec["tpot_s"] > 0
    assert fake.requests[-1]["temperature"] == 0.0 and fake.requests[-1]["stream"] is True  # температура явно
    assert fake.requests[-1]["stream_options"] == {"include_usage": True}
    assert fake.requests[-1]["cache_prompt"] is False


def test_client_failure_is_recorded_not_raised():
    from openai import OpenAI

    cfg = {"request": {"ignore_eos": True, "cache_prompt": False}, "endpoint": {"model": "m"},
           "generation": {"temperature": 0.0}}
    client = OpenAI(base_url="http://127.0.0.1:9/v1", api_key="none", max_retries=0, timeout=2)
    rec = run_request(client, cfg, {"id": "e", "condition": {}, "messages": [{"role": "user", "content": "hi"}],
                                    "max_tokens": 4})
    assert rec["status"] == "error" and rec["error"] and rec["ttft_s"] is None


def test_prompt_builder_hits_target_length(fake):
    from openai import OpenAI

    book, sha = load_book(ROOT / "data/pg1342.txt")
    assert len(sha) == 64 and "Project Gutenberg" not in book[:200]
    builder = PromptBuilder(ServerTokenizer(fake.root), book, "Summarize the following text in one sentence.")
    cfg = _cfg(fake)
    client = OpenAI(base_url=cfg["endpoint"]["base_url"], api_key="none", max_retries=0)
    ctx = Ctx.__new__(Ctx)
    ctx.cfg, ctx.builder, ctx.client, ctx.counter = cfg, builder, client, 0
    ctx.rd = SimpleNamespace(append_result=lambda r: None)
    ctx.calibrate()
    for target in (128, 1024, 4096):
        rec = run_request(client, cfg, {**ctx.request({}, target, 1), "extra_body": {"ignore_eos": False}})
        assert abs(rec["prompt_tokens"] - target) <= 1
    # метки уникальны и стоят в самом начале запроса
    m1, m2 = builder.build(1, 128), builder.build(2, 128)
    assert m1[0]["content"].startswith("[req-00001]") and m2[0]["content"].startswith("[req-00002]")


# ---------- каркас целиком: запуск, отказ при грязном дереве, пересчёт сводки ----------

def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)


@pytest.fixture()
def project(tmp_path, fake):
    for item in COPY_ITEMS:
        src = ROOT / item
        if src.is_dir():
            shutil.copytree(src, tmp_path / item, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy2(src, tmp_path / item)
    ep = yaml.safe_load((tmp_path / "endpoint.yaml").read_text(encoding="utf-8"))
    ep["endpoint"]["base_url"] = fake.root + "/v1"
    ep["endpoint"]["server_root"] = fake.root
    (tmp_path / "endpoint.yaml").write_text(yaml.safe_dump(ep), encoding="utf-8")
    small = {"experiment": "lr1", "series": "t_len", "kind": "prompt_length", "lengths": [64, 256], "repeats": 2,
             "max_tokens": 4, "expected_slots": 1, "max_background_cpu_pct": 100}
    (tmp_path / "configs/t_len.yaml").write_text(yaml.safe_dump(small), encoding="utf-8")
    load = {"experiment": "lr1", "series": "t_load", "kind": "load", "prompt_tokens": 200, "max_tokens": 8,
            "levels": [1, 2], "min_requests": 4, "requests_per_client": 2, "expected_slots": 1,
            "limits": {"ttft_p90_s": 5, "tpot_median_s": 1}, "max_background_cpu_pct": 100}
    (tmp_path / "configs/t_load.yaml").write_text(yaml.safe_dump(load), encoding="utf-8")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@e", "commit", "-q", "-m", "init")
    return tmp_path


def _run(project, *args):
    return subprocess.run([sys.executable, "run.py", *args], cwd=project, capture_output=True, text=True, encoding="utf-8",
                          env={**__import__("os").environ, "PYTHONUTF8": "1"})


def _newest(project, root="runs"):
    return sorted((project / root).iterdir())[-1]


def test_run_prompt_length_end_to_end(project):
    res = _run(project, "--config", "configs/t_len.yaml", "--allow-busy")
    assert res.returncode == 0, res.stderr[-2000:]
    d = _newest(project)
    meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
    assert meta["status"] == "completed" and len(meta["commit"]) == 40 and meta["dirty_files"] == []
    records = [json.loads(l) for l in (d / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    work = [r for r in records if not r["warmup"]]
    assert len(work) == 4 and len(records) - len(work) == 2          # прогрев и калибровка помечены, в статистику не входят
    assert all(r["completion_tokens"] == 4 for r in work)
    metrics_json = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    cfg = yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8"))
    again = json.loads(json.dumps(summarize(cfg, records)))           # сводка пересчитывается из results.jsonl
    assert metrics_json == again and metrics_json["length_mismatch"] == 0
    assert set(metrics_json["by_length"]) == {"64", "256"}
    assert metrics_json["by_length"]["256"]["ttft_s"]["median"] > metrics_json["by_length"]["64"]["ttft_s"]["median"]


def test_dirty_tree_refuses_final_run_but_not_debug(project):
    (project / "lr1/metrics.py").write_text((project / "lr1/metrics.py").read_text(encoding="utf-8") + "\n# x\n",
                                            encoding="utf-8")
    res = _run(project, "--config", "configs/t_len.yaml", "--allow-busy")
    assert res.returncode != 0 and "незакоммиченные изменения" in (res.stderr + res.stdout)
    res = _run(project, "--config", "configs/t_len.yaml", "--debug")
    assert res.returncode == 0, res.stderr[-2000:]
    assert (project / "runs_debug").exists() and not (project / "runs").exists()


def test_wrong_slot_count_aborts_before_any_request(project, fake):
    cfg = yaml.safe_load((project / "configs/t_len.yaml").read_text(encoding="utf-8"))
    cfg["expected_slots"] = 4
    (project / "configs/t_len.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    _git(project, "add", ".")
    _git(project, "-c", "user.name=t", "-c", "user.email=t@e", "commit", "-q", "-m", "slots")
    n_before = len(fake.requests)
    res = _run(project, "--config", "configs/t_len.yaml", "--allow-busy")
    assert res.returncode != 0 and "слотов" in (res.stderr + res.stdout)
    assert len(fake.requests) == n_before


def test_load_levels_throughput_grows_with_clients(project):
    res = _run(project, "--config", "configs/t_load.yaml", "--allow-busy")
    assert res.returncode == 0, res.stderr[-2000:]
    m = json.loads((_newest(project) / "metrics.json").read_text(encoding="utf-8"))
    one, two = m["levels"]["1"], m["levels"]["2"]
    assert one["n_requests"] == 4 and two["n_requests"] == 4 and one["n_error"] == two["n_error"] == 0
    assert two["throughput_tok_s"] > 1.4 * one["throughput_tok_s"]    # заглушка обслуживает клиентов параллельно
    assert one["wall_s"] > two["wall_s"]
