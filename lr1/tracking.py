"""Git, метаданные, каталог запуска и журнал MLflow (каркас ЛР0, адаптированный для ЛР1)."""
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import yaml

TRACKING_URI = "sqlite:///mlflow.db"


# ---------- git ----------

def _git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def git_commit():
    return _git("rev-parse", "HEAD")


def git_dirty():
    """Изменённые и новые неигнорируемые файлы (пусто — дерево чистое)."""
    return _git("status", "--porcelain")


def require_clean_tree():
    """Итоговый запуск только из закоммиченного кода."""
    dirty = git_dirty()
    if dirty:
        raise SystemExit(
            "Есть незакоммиченные изменения — итоговый запуск невозможен:\n" + dirty
            + "\n(для отладки добавьте --debug)"
        )
    return git_commit()


# ---------- окружение ----------

def library_versions():
    libs = ["openai", "httpx", "numpy", "pyyaml", "mlflow", "pandas", "matplotlib"]
    out = {"python": platform.python_version()}
    for lib in libs:
        try:
            out[lib] = version(lib)
        except Exception:  # noqa: BLE001
            out[lib] = None
    return out


def cpu_load_percent(samples=3):
    """Средняя загрузка всех процессоров за несколько секунд (Windows). None, если измерить не удалось.

    Нужна, чтобы замеры задержки не делались при посторонней нагрузке на процессор."""
    # WMI вместо Get-Counter: имена счётчиков Get-Counter зависят от языка Windows
    ps = ("1.." + str(samples) + " | ForEach-Object { "
          "(Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor -Filter \"Name='_Total'\").PercentProcessorTime; "
          "Start-Sleep -Milliseconds 700 }")
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=60)
        vals = [float(x.replace(",", ".")) for x in r.stdout.split() if x.strip()]
        return sum(vals) / len(vals) if vals else None
    except Exception:  # noqa: BLE001
        return None


def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_file(path, chunk=1 << 20):
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


# ---------- каталог запуска ----------

class RunDir:
    """runs/<run_id>/: config.yaml, metadata.json, results.jsonl, metrics.json."""

    def __init__(self, root, run_id):
        self.path = Path(root) / run_id
        self.path.mkdir(parents=True, exist_ok=False)
        self.metadata = {}
        self._results = open(self.path / "results.jsonl", "w", encoding="utf-8")

    def write_config(self, cfg):
        with open(self.path / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    def update_metadata(self, **fields):
        self.metadata.update(fields)
        (self.path / "metadata.json").write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    def append_result(self, rec):
        """Запись дописывается сразу: при сбое уже полученные результаты остаются на диске."""
        self._results.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._results.flush()
        os.fsync(self._results.fileno())

    def append_snapshot(self, snap):
        with open(self.path / "snapshots.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(snap, ensure_ascii=False) + "\n")

    def close(self):
        self._results.close()


def read_results(run_path):
    with open(Path(run_path) / "results.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------- MLflow ----------

def setup_mlflow(experiment):
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(experiment)
    return mlflow


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out
