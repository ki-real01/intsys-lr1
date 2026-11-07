"""Запуск llama-server, ожидание готовности, разбор журнала запуска и паспорт сервера."""
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

SERVER_EXE = Path("tools/llama/llama-server.exe")
MODEL = Path("models/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
STATE = Path("server/server.json")
ALIAS = "Qwen3-4B-Instruct-2507"


def build_command(slots, ctx, threads, port):
    """Команда запуска со всеми флагами (идёт в паспорт сервера)."""
    return [
        str(SERVER_EXE), "-m", str(MODEL), "-c", str(ctx), "-np", str(slots), "-t", str(threads),
        "-ngl", "0", "--host", "127.0.0.1", "--port", str(port), "-a", ALIAS, "--metrics",
    ]


def is_healthy(port):
    try:
        return httpx.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200
    except Exception:  # noqa: BLE001
        return False


def start(slots, ctx=16384, threads=6, port=8080, wait_s=600):
    if STATE.exists():
        raise SystemExit(f"Сервер уже запущен (см. {STATE}); сначала `python serve.py stop`")
    Path("server_logs").mkdir(exist_ok=True)
    Path("server").mkdir(exist_ok=True)
    log_path = Path("server_logs") / f"llama-np{slots}-{datetime.now():%Y%m%d-%H%M%S}.log"
    cmd = build_command(slots, ctx, threads, port)
    log = open(log_path, "w", encoding="utf-8")
    flags = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
    STATE.write_text(json.dumps({"pid": proc.pid, "port": port, "slots": slots, "ctx": ctx,
                                 "threads": threads, "log": str(log_path), "command": cmd},
                                ensure_ascii=False, indent=2), encoding="utf-8")
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < wait_s:
        if proc.poll() is not None:
            STATE.unlink(missing_ok=True)
            raise SystemExit(f"llama-server завершился с кодом {proc.returncode}; журнал: {log_path}")
        if is_healthy(port):
            return json.loads(STATE.read_text(encoding="utf-8"))
        time.sleep(1)
    stop()
    raise SystemExit(f"Сервер не стал готов за {wait_s} с; журнал: {log_path}")


def stop():
    if not STATE.exists():
        return False
    st = json.loads(STATE.read_text(encoding="utf-8"))
    subprocess.run(["taskkill", "/PID", str(st["pid"]), "/T", "/F"], capture_output=True)
    STATE.unlink(missing_ok=True)
    return True


def parse_log(log_path):
    """Достаёт из журнала запуска то, что нужно для паспорта и расчёта KV-кеша."""
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    def num(pattern, cast=int):
        m = re.search(pattern, text)
        return cast(m.group(1)) if m else None

    info = {
        "build": next((l.strip() for l in lines if l.startswith("build:") or "build_info" in l), None),
        "n_layer": num(r"print_info: n_layer\s*=\s*(\d+)"),
        "n_head": num(r"print_info: n_head\s*=\s*(\d+)"),
        "n_head_kv": num(r"print_info: n_head_kv\s*=\s*(\d+)"),
        "head_dim_k": num(r"print_info: n_embd_head_k\s*=\s*(\d+)"),
        "head_dim_v": num(r"print_info: n_embd_head_v\s*=\s*(\d+)"),
        "n_ctx_train": num(r"print_info: n_ctx_train\s*=\s*(\d+)"),
        "n_ctx": num(r"llama_context: n_ctx\s*=\s*(\d+)"),
        "n_ctx_seq": num(r"llama_context: n_ctx_seq\s*=\s*(\d+)"),
        "kv_size_mib": num(r"llama_kv_cache: size\s*=\s*([\d.]+)\s*MiB", float),
        "kv_cells": num(r"llama_kv_cache: size\s*=\s*[\d.]+\s*MiB\s*\(\s*(\d+)\s*cells"),
        "model_size_gib": num(r"print_info: file size\s*=\s*([\d.]+)\s*GiB", float),
        "model_params_b": num(r"print_info: model params\s*=\s*([\d.]+)\s*B", float),
        "n_slots": num(r"n_slots\s*=\s*(\d+)"),
    }
    keys = re.compile(r"(n_ctx|n_layer|n_head_kv|n_embd_head|kv_cache|KV|file size|model params|n_slots|"
                      r"new slot|build|system_info|n_threads|flash|unified|chat template|general\.name)", re.I)
    info["excerpt"] = [l for l in lines if keys.search(l)][:60]
    return info


def hardware():
    """Устройство, память, число потоков CPU (для паспорта)."""
    out = {"os": sys.platform, "cpu_count_logical": os.cpu_count()}
    try:
        ps = ("$c=Get-CimInstance Win32_Processor|Select-Object -First 1 Name,NumberOfCores,NumberOfLogicalProcessors;"
              "$m=[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,1);"
              "[pscustomobject]@{cpu=$c.Name.Trim();cores=$c.NumberOfCores;threads=$c.NumberOfLogicalProcessors;ram_gb=$m}|ConvertTo-Json")
        r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=60)
        out.update(json.loads(r.stdout))
    except Exception as exc:  # noqa: BLE001
        out["hardware_error"] = str(exc)
    return out


def passport(manifest_path="data/manifest.json"):
    """Паспорт запущенного сервера: модель, формат, версия, команда, контекст, KV-кеш, устройство."""
    st = json.loads(STATE.read_text(encoding="utf-8"))
    base = f"http://127.0.0.1:{st['port']}"
    props = httpx.get(f"{base}/props", timeout=30).json()
    models = httpx.get(f"{base}/v1/models", timeout=30).json()
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    ver = subprocess.run([str(SERVER_EXE), "--version"], capture_output=True, text=True)
    return {
        "model": manifest["models"]["qwen3_4b_q4km"],
        "server": {"name": "llama.cpp llama-server", "version": (ver.stdout + ver.stderr).strip().splitlines()[0],
                   "build_info": props.get("build_info"), "tool": manifest["tools"]["llama_cpp"]},
        "command": " ".join(st["command"]),
        "context_total": st["ctx"], "slots": st["slots"], "threads": st["threads"],
        "total_slots_reported": props.get("total_slots"),
        "api_models": [m.get("id") for m in models.get("data", [])],
        "log": parse_log(st["log"]),
        "hardware": hardware(),
        "device": "CPU (без GPU, -ngl 0)",
    }
