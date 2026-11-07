"""Один запуск эксперимента ЛР1 (клиент-измеритель, адаптер каркаса ЛР0).

    python run.py --config configs/exp1_prompt_length.yaml [--series NAME] [--debug]

Итоговый запуск (без --debug) требует чистого git-дерева и пишет в runs/ и эксперимент MLflow "lr1".
Отладочный (--debug) — в runs_debug/ и "lr1_debug"; в отчёт он не попадает.
Сервер должен быть уже запущен: `python serve.py start --slots N`.
"""
import argparse
import json
import os
import sys
import traceback
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

import httpx  # noqa: E402

from lr1 import server  # noqa: E402
from lr1.config import ConfigError, load_config  # noqa: E402
from lr1.experiments import RUNNERS, Ctx, summarize  # noqa: E402
from lr1.prompts import PromptBuilder, ServerTokenizer, load_book  # noqa: E402
from lr1.tracking import (  # noqa: E402
    RunDir, cpu_load_percent, flatten, git_commit, git_dirty, library_versions, now_iso, read_results, require_clean_tree,
    setup_mlflow,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="ЛР1: один запуск эксперимента")
    p.add_argument("--config", required=True)
    p.add_argument("--series", help="имя серии (по умолчанию из конфигурации)")
    p.add_argument("--debug", action="store_true", help="отладочный запуск: можно с грязным деревом")
    p.add_argument("--allow-busy", action="store_true",
                   help="не отказываться, если процессор занят посторонней нагрузкой (замеры будут искажены)")
    return p.parse_args(argv)


def server_info(cfg):
    """Что на самом деле запущено: слоты, модель, команда, KV-кеш из журнала запуска."""
    root = cfg["endpoint"]["server_root"]
    props = httpx.get(f"{root}/props", timeout=30).json()
    info = {"total_slots": props.get("total_slots"), "model_path": props.get("model_path"),
            "build_info": props.get("build_info")}
    if server.STATE.exists():
        st = json.loads(server.STATE.read_text(encoding="utf-8"))
        info.update(command=" ".join(st["command"]), ctx=st["ctx"], threads=st["threads"], log=st["log"],
                    log_info=server.parse_log(st["log"]))
    return info


def run(cfg, debug=False, allow_busy=False):
    commit, dirty = git_commit(), git_dirty()
    if not debug:
        require_clean_tree()

    book_text, book_sha = load_book(cfg["book"]["path"])
    manifest = json.loads(Path("data/manifest.json").read_text(encoding="utf-8"))
    expected = manifest["datasets"]["pride_and_prejudice"]["files"]["data/pg1342.txt"]["sha256"]
    if book_sha != expected:
        raise SystemExit(f"SHA-256 текста книги ({book_sha}) не совпадает с манифестом ({expected})")

    load_before = cpu_load_percent()
    limit = cfg.get("max_background_cpu_pct", 25)
    if not debug and not allow_busy and load_before is not None and load_before > limit:
        raise SystemExit(f"Процессор занят посторонней нагрузкой ({load_before:.0f} % > {limit} %): "
                         f"замеры задержки будут искажены. Дождитесь окончания нагрузки или добавьте --allow-busy")

    info = server_info(cfg)
    want = cfg.get("expected_slots")
    if want is not None and info["total_slots"] != want:
        raise SystemExit(f"Эксперимент требует {want} слотов, а сервер запущен с {info['total_slots']}: "
                         f"перезапустите `python serve.py stop` и `python serve.py start --slots {want}`")

    runs_root = "runs_debug" if debug else "runs"
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{cfg['series']}-{uuid.uuid4().hex[:6]}"
    rd = RunDir(runs_root, run_id)
    rd.write_config(cfg)
    rd.update_metadata(
        run_id=run_id, experiment=cfg["experiment"], series=cfg["series"], kind=cfg["kind"], status="running",
        debug=debug, started_at=now_iso(), finished_at=None, commit=commit,
        dirty_files=dirty.splitlines() if dirty else [], book_sha256=book_sha,
        model=cfg["endpoint"]["model"], generation=cfg["generation"], server=info,
        versions=library_versions(), command=" ".join(sys.argv), cpu_load_before_pct=load_before,
    )

    mlflow = setup_mlflow("lr1_debug" if debug else "lr1")
    with mlflow.start_run(run_name=run_id) as mlrun:
        mlflow.set_tags({"experiment": cfg["experiment"], "series": cfg["series"], "kind": cfg["kind"],
                         "commit": commit, "model": cfg["endpoint"]["model"], "run_id": run_id})
        mlflow.log_params(flatten({k: v for k, v in cfg.items() if k != "endpoint"}))
        rd.update_metadata(mlflow_run_id=mlrun.info.run_id)
        try:
            tokenizer = ServerTokenizer(cfg["endpoint"]["server_root"])
            builder = PromptBuilder(tokenizer, book_text, cfg["request"]["instruction"])
            rd.update_metadata(book_tokens=len(builder.book_tokens))
            ctx = Ctx(cfg, rd, builder)
            ctx.server_info = info
            overhead = ctx.calibrate()
            rd.update_metadata(prompt_overhead_tokens=overhead)

            RUNNERS[cfg["kind"]](ctx)
            rd.close()

            records = read_results(rd.path)
            snaps = []
            snap_path = rd.path / "snapshots.jsonl"
            if snap_path.exists():
                snaps = [json.loads(l) for l in snap_path.read_text(encoding="utf-8").splitlines() if l.strip()]
            summary = summarize(cfg, records, snaps)
            (rd.path / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            flat = {k: v for k, v in flatten(summary, "").items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
            mlflow.log_metrics({k.replace(" ", "_")[:240]: float(v) for k, v in flat.items()})
            rd.update_metadata(status="completed", finished_at=now_iso(), cpu_load_after_pct=cpu_load_percent(1))
        except BaseException as exc:  # noqa: BLE001 — включая KeyboardInterrupt: статус сохраняем
            rd.close()
            rd.update_metadata(status="failed", finished_at=now_iso(), error=f"{type(exc).__name__}: {exc}",
                               traceback=traceback.format_exc())
            mlflow.set_tag("status", "failed")
            mlflow.log_artifacts(str(rd.path))
            raise
        mlflow.set_tag("status", "completed")
        mlflow.log_artifacts(str(rd.path))
    return rd.path


def main(argv=None):
    args = parse_args(argv)
    overrides = {"series": args.series} if args.series else {}
    try:
        cfg = load_config(args.config, overrides, root=ROOT)
    except ConfigError as e:
        print(e, file=sys.stderr)
        return 2
    path = run(cfg, debug=args.debug, allow_busy=args.allow_busy)
    print(f"Запуск завершён: {path}\n{(path / 'metrics.json').read_text(encoding='utf-8')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
