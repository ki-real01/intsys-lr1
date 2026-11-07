"""Таблицы, графики и отчёт строятся только из сохранённых результатов.

    python make_report.py [--runs runs] [--out report]

Читает runs/*/ (metadata.json, results.jsonl, metrics.json), server/passport_np*.json и data/manifest.json.
Пишет report/*.csv, report/*.png, report/numbers.json и собирает report/REPORT.md из шаблона
report/REPORT_template.md (числа подставляются скриптом, а не вручную).
Рисунки и таблицы также сохраняются в MLflow (запись «report»).
"""
import argparse
import csv
import json
import os
import statistics
import sys
from pathlib import Path
from string import Template

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lr1 import metrics as M  # noqa: E402
from lr1.tracking import read_results, setup_mlflow  # noqa: E402

SERIES = ["checks", "exp1_prompt_length", "exp1_prefix_cache", "exp2_output_length", "exp3_load"]
RU = str.maketrans({".": ","})


def ru(x, nd=2):
    return f"{x:.{nd}f}".translate(RU)


def load_series(runs_dir):
    found = {}
    for d in sorted(Path(runs_dir).iterdir()):
        meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
        if meta["status"] != "completed" or meta["series"] not in SERIES:
            continue
        if meta["series"] in found:
            raise SystemExit(f"Серия {meta['series']} выполнена больше одного раза "
                             f"({found[meta['series']]['dir'].name}, {d.name}): выбирать удачный запуск нельзя. "
                             f"Уберите лишний каталог из {runs_dir} или объясните повтор в отчёте.")
        found[meta["series"]] = {"dir": d, "meta": meta,
                                 "metrics": json.loads((d / "metrics.json").read_text(encoding="utf-8"))}
    return found


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def memory_numbers(passports, manifest):
    """Расчёт памяти весов и KV-кеша по параметрам модели из журнала сервера."""
    p = passports[min(passports)]["log"]
    layers, heads, dim = p["n_layer"], p["n_head_kv"], p["head_dim_k"]
    per_tok = M.kv_bytes_per_token(layers, heads, dim, 2)
    params = p["model_params_b"] * 1e9
    gguf = manifest["models"]["qwen3_4b_q4km"]["bytes"]
    n_ctx_total = passports[max(passports)]["context_total"]
    return {
        "n_layer": layers, "n_head_kv": heads, "head_dim": dim, "params_b": p["model_params_b"],
        "kv_bytes_per_token": per_tok, "kv_kib_per_token": per_tok / 1024,
        "weights_fp16_gb": params * 2 / 1e9, "weights_fp16_gib": params * 2 / 2**30,
        "weights_gguf_gb": gguf / 1e9, "weights_gguf_gib": gguf / 2**30,
        "kv_ctx_total_tokens": n_ctx_total, "kv_ctx_total_mib": n_ctx_total * per_tok / 2**20,
        "kv_8x4000_gib": 8 * 4000 * per_tok / 2**30, "kv_32768_gib": 32768 * per_tok / 2**30,
        "kv_per_request_1152_mib": 1152 * per_tok / 2**20,
        "requests_1152_in_total_ctx": n_ctx_total // 1152,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="report")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    S = load_series(args.runs)
    missing = [s for s in SERIES if s not in S]
    if missing:
        print("ВНИМАНИЕ: нет завершённых запусков серий:", ", ".join(missing))
    manifest = json.loads(Path("data/manifest.json").read_text(encoding="utf-8"))
    passports = {}
    for path in sorted(Path("server").glob("passport_np*.json")):
        pp = json.loads(path.read_text(encoding="utf-8"))
        passports[pp["slots"]] = pp
    N = {"series": {s: S[s]["meta"]["run_id"] for s in S}}
    N["commit"] = {s: S[s]["meta"]["commit"][:7] for s in S}
    N["cpu_load_before"] = {s: S[s]["meta"].get("cpu_load_before_pct") for s in S}
    N["cpu_load_after"] = {s: S[s]["meta"].get("cpu_load_after_pct") for s in S}
    tables = {}
    if passports:
        N["memory"] = memory_numbers(passports, manifest)
        N["server"] = {k: {"command": v["command"], "version": v["server"]["version"], "ctx_total": v["context_total"],
                           "slots": v["slots"], "threads": v["threads"], "kv_size_mib": v["log"]["kv_size_mib"],
                           "n_ctx_seq": v["log"]["n_ctx_seq"], "kv_cells": v["log"]["kv_cells"]}
                       for k, v in passports.items()}
        N["hardware"] = passports[min(passports)]["hardware"]

    # ---- эксперимент 1: длина запроса
    if "exp1_prompt_length" in S:
        m = S["exp1_prompt_length"]["metrics"]
        rows, xs, tt_med, tt_lo, tt_hi, sp_med, sp_lo, sp_hi = [], [], [], [], [], [], [], []
        for length, e in m["by_length"].items():
            t, sp, pt = e["ttft_s"], e["prefill_tok_s"], e["prompt_tokens"]
            rows.append([length, e["n"], ru(pt["median"], 0), ru(t["median"]), f"{ru(t['min'])}–{ru(t['max'])}",
                         ru(sp["median"], 1), f"{ru(sp['min'], 1)}–{ru(sp['max'], 1)}", e["n_error"]])
            xs.append(pt["median"])
            tt_med.append(t["median"]); tt_lo.append(t["min"]); tt_hi.append(t["max"])
            sp_med.append(sp["median"]); sp_lo.append(sp["min"]); sp_hi.append(sp["max"])
        write_csv(out / "exp1_by_length.csv", ["target_tokens", "n", "prompt_tokens_median", "ttft_median_s",
                                               "ttft_min_max_s", "prefill_tok_s_median", "prefill_min_max", "errors"], rows)
        tables["exp1_table"] = md_table(["Длина запроса, токенов", "Запросов", "Токенов по usage", "TTFT, медиана, с",
                                         "TTFT, мин–макс, с", "Скорость prefill, ток/с", "мин–макс", "Ошибок"], rows)
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
        ax[0].errorbar(xs, tt_med, yerr=[[a - b for a, b in zip(tt_med, tt_lo)], [b - a for a, b in zip(tt_med, tt_hi)]],
                       fmt="o-", capsize=3)
        ax[0].set_xlabel("длина запроса, токенов"); ax[0].set_ylabel("TTFT, с"); ax[0].set_title("TTFT от длины запроса")
        ax[1].errorbar(xs, sp_med, yerr=[[a - b for a, b in zip(sp_med, sp_lo)], [b - a for a, b in zip(sp_med, sp_hi)]],
                       fmt="s-", color="tab:orange", capsize=3)
        ax[1].set_xlabel("длина запроса, токенов"); ax[1].set_ylabel("токенов/с"); ax[1].set_title("Скорость prefill")
        ax[1].set_ylim(bottom=0)
        for a in ax:
            a.grid(alpha=0.3)
        fig.suptitle("Эксперимент 1: медиана и диапазон по 5 запросам", fontsize=10)
        fig.tight_layout(); fig.savefig(out / "fig_exp1_ttft.png", dpi=150); plt.close(fig)
        speeds = sp_med
        N["exp1"] = {"ttft_by_len": {k: e["ttft_s"]["median"] for k, e in m["by_length"].items()},
                     "prefill_by_len": {k: e["prefill_tok_s"]["median"] for k, e in m["by_length"].items()},
                     "prefill_median": sp_med,
                     "prefill_first": speeds[0], "prefill_last": speeds[-1], "prefill_peak": max(speeds),
                     "prefill_last_to_peak": speeds[-1] / max(speeds),
                     "ttft_ratio_last_first": tt_med[-1] / tt_med[0], "length_ratio_last_first": xs[-1] / xs[0],
                     "length_mismatch": m.get("length_mismatch"),
                     "max_prompt_deviation": max(e["max_prompt_deviation"] for e in m["by_length"].values())}

    if "exp1_prefix_cache" in S:
        m = S["exp1_prefix_cache"]["metrics"]
        rows = [[a["attempt"] + 1, ru(a["ttft_s"]), a["prompt_tokens"], a.get("cache_n"), a.get("prompt_n")] for a in m["attempts"]]
        write_csv(out / "exp1_prefix_cache.csv", ["attempt", "ttft_s", "prompt_tokens", "cache_n", "prompt_n"], rows)
        tables["prefix_table"] = md_table(["Попытка", "TTFT, с", "Токенов запроса", "Из кеша, токенов", "Обработано, токенов"], rows)
        at = m["attempts"]
        N["prefix"] = {"t1": at[0]["ttft_s"], "t2": at[1]["ttft_s"], "t3": at[2]["ttft_s"],
                       "ratio_1_2": m.get("ttft_ratio_1_to_2"), "ratio_1_3": m.get("ttft_ratio_1_to_3"),
                       "cache_first": at[0].get("cache_n"), "cache_second": at[1].get("cache_n"),
                       "prompt_tokens": at[0]["prompt_tokens"]}

    # ---- эксперимент 2: длина ответа
    if "exp2_output_length" in S:
        m = S["exp2_output_length"]["metrics"]
        res = read_results(S["exp2_output_length"]["dir"])
        ok = [r for r in res if not r["warmup"] and r["status"] == "ok"]
        rows = []
        for n_out, e in m["by_output"].items():
            rows.append([n_out, e["n"], ru(e["completion_tokens"]["median"], 0), ru(e["e2e_s"]["median"]),
                         f"{ru(e['e2e_s']['min'])}–{ru(e['e2e_s']['max'])}", ru(e["ttft_s"]["median"]),
                         ru(e["tpot_s"]["median"] * 1000, 0), e["n_error"]])
        write_csv(out / "exp2_by_output.csv", ["output_tokens_target", "n", "completion_tokens", "e2e_median_s",
                                               "e2e_min_max_s", "ttft_median_s", "tpot_median_ms", "errors"], rows)
        tables["exp2_table"] = md_table(["Длина ответа, токенов", "Запросов", "Токенов по usage", "Полное время, медиана, с",
                                         "мин–макс, с", "TTFT, медиана, с", "TPOT, медиана, мс", "Ошибок"], rows)
        fit = m["fit_e2e_all_requests"]
        tpot_all = [r["tpot_s"] for r in ok]
        ttft_all = [r["ttft_s"] for r in ok]
        fig, ax = plt.subplots(figsize=(6.4, 4))
        ax.scatter([r["completion_tokens"] for r in ok], [r["e2e_s"] for r in ok], s=18, alpha=0.7, label="запросы")
        xs = [0, max(r["completion_tokens"] for r in ok)]
        ax.plot(xs, [fit["intercept"] + fit["slope"] * x for x in xs], "r-",
                label=f"прямая: {ru(fit['intercept'], 1)} с + {ru(fit['slope'] * 1000, 0)} мс × N")
        ax.set_xlabel("длина ответа N, токенов"); ax.set_ylabel("полное время ответа, с")
        ax.set_title("Эксперимент 2: полное время от длины ответа"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out / "fig_exp2_e2e.png", dpi=150); plt.close(fig)
        N["exp2"] = {"slope_s": fit["slope"], "slope_ms": fit["slope"] * 1000, "intercept_s": fit["intercept"], "r2": fit["r2"],
                     "tpot_median_ms": statistics.median(tpot_all) * 1000, "tpot_min_ms": min(tpot_all) * 1000,
                     "tpot_max_ms": max(tpot_all) * 1000, "ttft_median_s": statistics.median(ttft_all),
                     "tok_s": 1 / statistics.median(tpot_all), "length_mismatch": m.get("length_mismatch"),
                     "by_output": {k: {"e2e_median": v["e2e_s"]["median"], "tpot_median_ms": v["tpot_s"]["median"] * 1000}
                                   for k, v in m["by_output"].items()}}
        if "memory" in N:
            bw = 51.2e9  # двухканальная DDR4-3200: 2 модуля по 3200 MT/s × 8 байт × 2 канала (предположение)
            wb = manifest["models"]["qwen3_4b_q4km"]["bytes"]
            N["exp2"]["bandwidth_gb_s_assumed"] = bw / 1e9
            N["exp2"]["tpot_lower_bound_ms"] = wb / bw * 1000
            N["exp2"]["tok_s_upper_bound"] = bw / wb
            N["exp2"]["tpot_over_bound"] = N["exp2"]["tpot_median_ms"] / N["exp2"]["tpot_lower_bound_ms"]
            N["exp2"]["effective_bandwidth_gb_s"] = wb / statistics.median(tpot_all) / 1e9

    # ---- эксперимент 3: нагрузка
    if "exp3_load" in S:
        m = S["exp3_load"]["metrics"]
        lim = m["limits"]
        # метрики сервера: в этой сборке llama-server не отдаёт заполнение KV-кеша, но отдаёт максимум токенов в слоте
        snaps = [json.loads(l) for l in (S["exp3_load"]["dir"] / "snapshots.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        n_ctx_slot = N["server"][max(N["server"])]["n_ctx_seq"] if "server" in N else None
        max_tok = {}
        for sn in snaps:
            v = sn["metrics"].get("llamacpp:n_tokens_max")
            if v is not None:
                max_tok[str(sn["clients"])] = max(max_tok.get(str(sn["clients"]), 0), v)
        rows, xs, thr, tp_med, tt_p90 = [], [], [], [], []
        for n_cl, e in m["levels"].items():
            rows.append([n_cl, e["n_requests"], ru(e["wall_s"], 0), ru(e["throughput_tok_s"]), ru(e["requests_per_s"], 3),
                         ru(e["ttft_median_s"], 1), ru(e["ttft_p90_s"], 1), ru(e["tpot_median_s"] * 1000, 0),
                         ru(e["goodput_req_s"], 3), int(e.get("max_requests_deferred") or 0),
                         f"{max_tok[n_cl]:.0f}" if n_cl in max_tok else "—",
                         "да" if e["within_limits"] else "нет"])
            xs.append(int(n_cl)); thr.append(e["throughput_tok_s"]); tp_med.append(e["tpot_median_s"] * 1000)
            tt_p90.append(e["ttft_p90_s"])
        hdr = ["Клиентов", "Запросов", "Время уровня, с", "Пропускная способность, ток/с", "Запросов/с",
               "TTFT, медиана, с", "TTFT, p90, с", "TPOT, медиана, мс", "Goodput, запр./с",
               "Макс. в очереди", "Макс. токенов в слоте", "В пределах"]
        write_csv(out / "exp3_levels.csv", hdr, rows)
        tables["exp3_table"] = md_table(hdr, rows)
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
        ax[0].plot(thr, tp_med, "o-")
        ax[0].axhline(lim["tpot_median_s"] * 1000, color="r", ls="--", label=f"предел {ru(lim['tpot_median_s'] * 1000, 0)} мс")
        ax[1].plot(thr, tt_p90, "s-", color="tab:orange")
        ax[1].axhline(lim["ttft_p90_s"], color="r", ls="--", label=f"предел {ru(lim['ttft_p90_s'], 0)} с")
        for a, ys in ((ax[0], tp_med), (ax[1], tt_p90)):
            for x, t, y in zip(xs, thr, ys):
                a.annotate(f"{x} кл.", (t, y), textcoords="offset points", xytext=(4, 5), fontsize=8)
            a.set_xlabel("пропускная способность, выходных токенов/с"); a.grid(alpha=0.3); a.legend(fontsize=8)
            a.set_ylim(bottom=0)
        ax[0].set_ylabel("TPOT, медиана, мс"); ax[1].set_ylabel("TTFT, 90-й процентиль, с")
        fig.suptitle("Эксперимент 3: пропускная способность и задержка по уровням нагрузки", fontsize=10)
        fig.tight_layout(); fig.savefig(out / "fig_exp3_load.png", dpi=150); plt.close(fig)
        # разбор: из чего складывается время уровня. Prefill разных запросов идёт по очереди (вычисления),
        # поэтому его суммарное время берём как число запросов × TTFT одного клиента; остальное — время decode.
        lv1 = m["levels"][min(m["levels"], key=int)]
        t_pf = lv1["ttft_median_s"]
        lv = {}
        dec1 = None
        for n_cl, e in m["levels"].items():
            pf_total = e["n_requests"] * t_pf
            dec = e["wall_s"] - pf_total
            if dec1 is None:
                dec1 = dec
            n_decode_tokens = sum(r["completion_tokens"] - 1 for r in read_results(S["exp3_load"]["dir"])
                                  if not r["warmup"] and r["status"] == "ok" and r["condition"]["clients"] == int(n_cl))
            lv[n_cl] = {"prefill_total_s": pf_total, "prefill_share": pf_total / e["wall_s"], "decode_part_s": dec,
                        "decode_speedup": dec1 / dec, "decode_ms_per_token": dec / n_decode_tokens * 1000,
                        "tpot_median_ms": e["tpot_median_s"] * 1000, "ttft_p90_s": e["ttft_p90_s"],
                        "ttft_median_s": e["ttft_median_s"], "throughput": e["throughput_tok_s"],
                        "rps": e["requests_per_s"], "goodput": e["goodput_req_s"], "n_good": e["n_good"],
                        "wall_s": e["wall_s"], "n_requests": e["n_requests"], "goodput_s": ru(e["goodput_req_s"], 3), "max_tokens_slot": max_tok.get(n_cl),
                        "slot_fill_pct": (max_tok[n_cl] / n_ctx_slot * 100) if n_cl in max_tok and n_ctx_slot else None}
        good = [int(k) for k, e in m["levels"].items() if e["within_limits"]]
        N["exp3"] = {"limits": lim, "tpot_limit_ms": lim["tpot_median_s"] * 1000, "ttft_limit_s": lim["ttft_p90_s"], "lv": lv, "t_prefill_single": t_pf, "levels": {k: v for k, v in m["levels"].items()},
                     "max_clients_within_limits": max(good) if good else None,
                     "throughput_gain_last_first": thr[-1] / thr[0] if thr[0] else None,
                     "tpot_ratio_last_first": tp_med[-1] / tp_med[0], "ttft_ratio_last_first": tt_p90[-1] / tt_p90[0]}

    if "checks" in S:
        N["checks"] = S["checks"]["metrics"]

    (out / "numbers.json").write_text(json.dumps(N, ensure_ascii=False, indent=2), encoding="utf-8")

    template = out / "REPORT_template.md"
    if template.exists():
        class T(Template):
            delimiter = "$"
        flat = {}

        def put(prefix, obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    put(f"{prefix}_{k}" if prefix else str(k), v)
            elif isinstance(obj, float):
                # число знаков по величине: ≥100 — целые, ≥10 — один знак, иначе два; десятичная запятая
                flat[prefix] = ru(obj, 0 if abs(obj) >= 100 else 1 if abs(obj) >= 10 else 2)
            else:
                flat[prefix] = obj
        put("", N)
        flat.update(tables)
        (out / "REPORT.md").write_text(T(template.read_text(encoding="utf-8")).substitute(flat), encoding="utf-8")
        print("Собран report/REPORT.md")

    mlflow = setup_mlflow("lr1")
    for old in mlflow.search_runs(experiment_names=["lr1"], filter_string="tags.role = 'report'", output_format="list"):
        mlflow.delete_run(old.info.run_id)
    with mlflow.start_run(run_name="report"):
        mlflow.set_tag("role", "report")
        mlflow.log_artifacts(str(out), artifact_path="report")
    print(json.dumps(N, ensure_ascii=False, indent=1)[:3000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
