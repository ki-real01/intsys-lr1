"""Эксперименты ЛР1: длина запроса, кеш префикса, длина ответа, нагрузка.

Каждая функция пишет записи в results.jsonl по мере получения. Сводки считаются отдельно
функцией summarize() из записей results.jsonl (Правила, п. 4.2).
"""
import asyncio
import time

import httpx
from openai import AsyncOpenAI, OpenAI

from . import metrics
from .client import arun_request, run_request

PROBE_TEXT_TOKENS = 8
STRIDE = 5003  # шаг по тексту книги между запросами с «разным текстом»


# ---------- метрики сервера ----------

def parse_prometheus(text):
    """Все метрики llamacpp:* из ответа /metrics: {имя: число}."""
    out = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.startswith("llamacpp:"):
            continue
        name, _, value = line.rpartition(" ")
        try:
            out[name.strip()] = float(value)
        except ValueError:
            pass
    return out


def server_metrics(http):
    r = http.get("/metrics", timeout=10)
    r.raise_for_status()
    return parse_prometheus(r.text)


# ---------- общий контекст запуска ----------

class Ctx:
    def __init__(self, cfg, rd, builder):
        self.cfg, self.rd, self.builder = cfg, rd, builder
        ep = cfg["endpoint"]
        self.client = OpenAI(base_url=ep["base_url"], api_key=ep.get("api_key", "none"),
                             timeout=ep["timeout_s"], max_retries=0)  # повторы скрыли бы ошибки и исказили время
        self.http = httpx.Client(base_url=ep["server_root"], timeout=30)
        self.counter = 0

    def next_index(self):
        self.counter += 1
        return self.counter

    def request(self, condition, target_tokens, max_tokens, offset=0, warmup=False, cache_prompt=None, index=None):
        idx = self.next_index() if index is None else index
        extra = {} if cache_prompt is None else {"cache_prompt": cache_prompt}
        return {"id": f"{self.cfg['series']}-{idx:05d}", "index": idx, "condition": condition,
                "messages": self.builder.build(idx, target_tokens, offset),
                "max_tokens": max_tokens, "warmup": warmup, "extra_body": extra}

    def record(self, rec, req):
        rec["index"] = req["index"]
        self.rd.append_result(rec)
        flag = "прогрев " if rec["warmup"] else ""
        if rec["status"] == "ok":
            print(f"  {flag}{rec['id']} {rec['condition']}: вход {rec['prompt_tokens']}, выход {rec['completion_tokens']}, "
                  f"TTFT {rec['ttft_s']:.2f} с, e2e {rec['e2e_s']:.2f} с", flush=True)
        else:
            print(f"  {flag}{rec['id']} {rec['condition']}: ОШИБКА {rec['error']}", flush=True)
        return rec

    def send(self, req):
        return self.record(run_request(self.client, self.cfg, req), req)

    def calibrate(self):
        """Пробный запрос: накладные токены метки, инструкции и шаблона диалога."""
        idx = self.next_index()
        req = {"id": f"{self.cfg['series']}-calib-{idx:05d}", "index": idx, "condition": {"kind": "calibration"},
               "messages": self.builder.messages(idx, PROBE_TEXT_TOKENS, 0), "max_tokens": 1,
               "warmup": True, "extra_body": {"ignore_eos": False}}
        rec = self.send(req)
        if rec["status"] != "ok":
            raise RuntimeError(f"Калибровочный запрос не удался: {rec['error']}")
        return self.builder.calibrate(rec["prompt_tokens"], PROBE_TEXT_TOKENS)

    def warmup(self, condition, target_tokens, max_tokens):
        """Один прогревочный запрос перед серией; в статистику не входит."""
        return self.send(self.request({**condition, "kind": "warmup"}, target_tokens, max_tokens, warmup=True))


# ---------- эксперимент 1: длина запроса ----------

def run_prompt_length(ctx):
    cfg = ctx.cfg
    ctx.warmup({}, min(cfg["lengths"]), cfg["max_tokens"])
    for length in cfg["lengths"]:
        print(f"Длина запроса {length} токенов", flush=True)
        for rep in range(cfg["repeats"]):
            req = ctx.request({"prompt_tokens_target": length, "repeat": rep}, length, cfg["max_tokens"])
            ctx.send(req)


def run_prefix_cache(ctx):
    cfg = ctx.cfg
    length = cfg["prompt_tokens"]
    ctx.warmup({"prompt_tokens_target": length}, 128, cfg["max_tokens"])
    idx = ctx.next_index()  # одна и та же метка и один и тот же текст во всех попытках
    for attempt in range(cfg["times"]):
        req = ctx.request({"prompt_tokens_target": length, "attempt": attempt}, length, cfg["max_tokens"],
                          cache_prompt=True, index=idx)
        req["id"] = f"{cfg['series']}-{idx:05d}-try{attempt}"
        ctx.send(req)


# ---------- эксперимент 2: длина ответа ----------

def run_output_length(ctx):
    cfg = ctx.cfg
    ctx.warmup({}, cfg["prompt_tokens"], min(cfg["output_lengths"]))
    for n_out in cfg["output_lengths"]:
        print(f"Длина ответа {n_out} токенов", flush=True)
        for rep in range(cfg["repeats"]):
            req = ctx.request({"prompt_tokens_target": cfg["prompt_tokens"], "output_tokens_target": n_out,
                               "repeat": rep}, cfg["prompt_tokens"], n_out)
            ctx.send(req)


# ---------- эксперимент 3: нагрузка ----------

def n_requests_for_level(cfg, n_clients):
    return max(cfg["min_requests"], cfg["requests_per_client"] * n_clients)


async def _run_level(ctx, level):
    cfg = ctx.cfg
    ep = cfg["endpoint"]
    n_clients = level["clients"]
    n_text = ctx.builder.text_tokens_for(cfg["prompt_tokens"])
    span = len(ctx.builder.book_tokens) - n_text
    queue = asyncio.Queue()
    for i in range(level["n_requests"]):
        idx = ctx.next_index()
        offset = (idx * STRIDE) % span  # разный текст у разных запросов
        queue.put_nowait(ctx.request({"clients": n_clients, "prompt_tokens_target": cfg["prompt_tokens"]},
                                     cfg["prompt_tokens"], cfg["max_tokens"], offset=offset, index=idx))
    aclient = AsyncOpenAI(base_url=ep["base_url"], api_key=ep.get("api_key", "none"),
                          timeout=ep["timeout_s"], max_retries=0)

    async def worker(worker_id):
        while True:
            try:
                req = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            rec = await arun_request(aclient, cfg, req)
            rec["client"] = worker_id
            ctx.record(rec, req)  # запись в results.jsonl по мере получения

    async def sampler():
        async with httpx.AsyncClient(base_url=ep["server_root"], timeout=10) as http:
            while True:
                try:
                    r = await http.get("/metrics")
                    ctx.rd.append_snapshot({"clients": n_clients, "kind": "sample", "t_s": time.perf_counter(),
                                            "metrics": parse_prometheus(r.text)})
                except Exception:  # noqa: BLE001
                    pass
                await asyncio.sleep(2)

    sampler_task = asyncio.create_task(sampler())
    try:
        await asyncio.gather(*[worker(w) for w in range(n_clients)])
    finally:
        sampler_task.cancel()
        await aclient.close()


def run_load(ctx):
    cfg = ctx.cfg
    for n_clients in cfg["levels"]:
        n_req = n_requests_for_level(cfg, n_clients)
        print(f"Уровень нагрузки: {n_clients} клиентов, {n_req} запросов", flush=True)
        ctx.warmup({"clients": n_clients}, cfg["prompt_tokens"], cfg["max_tokens"])
        ctx.rd.append_snapshot({"clients": n_clients, "kind": "before", "t_s": time.perf_counter(),
                                "metrics": server_metrics(ctx.http)})
        asyncio.run(_run_level(ctx, {"clients": n_clients, "n_requests": n_req}))
        ctx.rd.append_snapshot({"clients": n_clients, "kind": "after", "t_s": time.perf_counter(),
                                "metrics": server_metrics(ctx.http)})


# ---------- проверки реализации на живом сервере (п. 3.4) ----------

def run_checks(ctx):
    """Три проверки, которым нужен сервер: точная длина ответа, повтор при температуре 0, KV-кеш."""
    cfg = ctx.cfg
    # 1. max_tokens = 64 и ignore_eos возвращают ровно 64 токена
    req = ctx.request({"check": "ignore_eos_exact", "prompt_tokens_target": 256}, 256, 64)
    ctx.send(req)
    # 2. один и тот же запрос при температуре 0, дважды подряд: совпали ли тексты
    idx = ctx.next_index()
    for attempt in range(2):
        req = ctx.request({"check": "temperature_zero_repeat", "attempt": attempt, "prompt_tokens_target": 256},
                          256, 128, index=idx)
        req["id"] = f"{cfg['series']}-{idx:05d}-try{attempt}"
        ctx.send(req)
    # 3. ёмкость KV-кеша из журнала сервера x байт на токен = размер KV-кеша из журнала
    li = (getattr(ctx, "server_info", {}) or {}).get("log_info", {})
    row = {"check": "kv_cache_size", "log": li}
    if all(li.get(k) for k in ("n_layer", "n_head_kv", "head_dim_k", "kv_cells", "kv_size_mib")):
        per_token = metrics.kv_bytes_per_token(li["n_layer"], li["n_head_kv"], li["head_dim_k"], 2)
        row.update(bytes_per_token=per_token, expected_mib=li["kv_cells"] * per_token / 2**20,
                   ratio=li["kv_size_mib"] / (li["kv_cells"] * per_token / 2**20))
    ctx.rd.append_snapshot({"clients": 0, "kind": "check", "metrics": row})


RUNNERS = {"prompt_length": run_prompt_length, "prefix_cache": run_prefix_cache,
           "output_length": run_output_length, "load": run_load, "checks": run_checks}


# ---------- сводки из results.jsonl ----------

def _stats(records, key):
    vals = [r[key] for r in records if r["status"] == "ok" and r.get(key) is not None]
    return metrics.median_range(vals) if vals else None


def summarize(cfg, records, snapshots=()):
    """Все итоговые показатели — только из записей results.jsonl (+ снимки метрик сервера)."""
    kind = cfg["kind"]
    work = [r for r in records if not r["warmup"]]
    out = {"kind": kind, "n_records": len(work), "n_warmup": len(records) - len(work),
           "n_ok": sum(r["status"] == "ok" for r in work), "n_error": sum(r["status"] != "ok" for r in work)}
    ignore_eos = cfg["request"]["ignore_eos"]
    if ignore_eos:
        out["length_mismatch"] = sum(1 for r in work if r["status"] == "ok" and r["completion_tokens"] != r["max_tokens"])

    if kind == "prompt_length":
        by_len = {}
        for r in work:
            by_len.setdefault(r["condition"]["prompt_tokens_target"], []).append(r)
        out["by_length"] = {}
        for length, recs in sorted(by_len.items()):
            ok = [r for r in recs if r["status"] == "ok"]
            entry = {"n": len(recs), "n_error": len(recs) - len(ok), "ttft_s": _stats(recs, "ttft_s"),
                     "prompt_tokens": _stats(recs, "prompt_tokens")}
            if ok:
                speeds = [r["prompt_tokens"] / r["ttft_s"] for r in ok]
                entry["prefill_tok_s"] = metrics.median_range(speeds)
                entry["max_prompt_deviation"] = max(abs(r["prompt_tokens"] - length) for r in ok)
            out["by_length"][length] = entry
    elif kind == "prefix_cache":
        by_try = sorted(work, key=lambda r: r["condition"]["attempt"])
        out["attempts"] = [{"attempt": r["condition"]["attempt"], "ttft_s": r["ttft_s"], "prompt_tokens": r["prompt_tokens"],
                            "cache_n": (r.get("timings") or {}).get("cache_n"),
                            "prompt_n": (r.get("timings") or {}).get("prompt_n")} for r in by_try if r["status"] == "ok"]
        t = [a["ttft_s"] for a in out["attempts"]]
        if len(t) >= 2:
            out["ttft_ratio_1_to_2"] = t[0] / t[1]
        if len(t) >= 3:
            out["ttft_ratio_1_to_3"] = t[0] / t[2]
    elif kind == "output_length":
        by_n = {}
        for r in work:
            by_n.setdefault(r["condition"]["output_tokens_target"], []).append(r)
        out["by_output"] = {n: {"n": len(recs), "n_error": sum(x["status"] != "ok" for x in recs),
                                "e2e_s": _stats(recs, "e2e_s"), "ttft_s": _stats(recs, "ttft_s"),
                                "tpot_s": _stats(recs, "tpot_s"), "completion_tokens": _stats(recs, "completion_tokens")}
                            for n, recs in sorted(by_n.items())}
        ok = [r for r in work if r["status"] == "ok"]
        if len({r["completion_tokens"] for r in ok}) >= 2:
            out["fit_e2e_all_requests"] = metrics.fit_line([r["completion_tokens"] for r in ok], [r["e2e_s"] for r in ok])
            med = [(v["completion_tokens"]["median"], v["e2e_s"]["median"]) for v in out["by_output"].values()
                   if v["e2e_s"] and v["completion_tokens"]]
            out["fit_e2e_medians"] = metrics.fit_line([m[0] for m in med], [m[1] for m in med])
    elif kind == "checks":
        by = {}
        for r in work:
            by.setdefault(r["condition"]["check"], []).append(r)
        exact = by.get("ignore_eos_exact", [])
        out["ignore_eos_exact"] = {"completion_tokens": [r["completion_tokens"] for r in exact],
                                   "passed": bool(exact) and all(r["completion_tokens"] == 64 for r in exact)}
        rep = sorted(by.get("temperature_zero_repeat", []), key=lambda r: r["condition"]["attempt"])
        out["temperature_zero_repeat"] = {"texts_equal": len(rep) == 2 and rep[0]["text"] == rep[1]["text"],
                                          "completion_tokens": [r["completion_tokens"] for r in rep]}
        kv = [s["metrics"] for s in snapshots if s.get("kind") == "check"]
        if kv:
            out["kv_cache_size"] = {k: v for k, v in kv[0].items() if k != "log"}
            out["kv_cache_size"]["passed"] = bool(kv[0].get("ratio")) and abs(kv[0]["ratio"] - 1) < 0.02
    elif kind == "load":
        lim = cfg["limits"]
        out["limits"] = lim
        levels = {}
        for r in work:
            levels.setdefault(r["condition"]["clients"], []).append(r)
        out["levels"] = {}
        for n_clients, recs in sorted(levels.items()):
            t_start = min(r["t0_s"] for r in recs)
            t_end = max(r["t_end_s"] for r in recs if r["t_end_s"] is not None)
            summary = metrics.level_summary(recs, t_end - t_start, lim["ttft_p90_s"], lim["tpot_median_s"])
            samples = [s for s in snapshots if s["clients"] == n_clients]
            for key, name in (("max_requests_deferred", "llamacpp:requests_deferred"),
                              ("max_requests_processing", "llamacpp:requests_processing"),
                              ("max_kv_cache_usage_ratio", "llamacpp:kv_cache_usage_ratio")):
                vals = [s["metrics"][name] for s in samples if name in s["metrics"]]
                summary[key] = max(vals) if vals else None
            out["levels"][n_clients] = summary
    return out
