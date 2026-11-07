# ЛР1. Инференс и развёртывание языковой модели

Курс «Интеллектуальные системы», задание: [dm780009/intsys](https://github.com/dm780009/intsys), `LR1_llm_inference_serving.pdf`.

**Вопрос работы:** из чего складывается время ответа языковой модели и как оно меняется, когда пользователей становится много?

Профиль — **CPU** (llama.cpp, `llama-server`), модель Qwen3-4B-Instruct-2507 в формате GGUF Q4_K_M.
Прогноз — [hypothesis.md](hypothesis.md), отчёт — [report/REPORT.md](report/REPORT.md).
Сервер и клиент-измеритель служат моделью для ЛР2–ЛР8: адрес и параметры по умолчанию — [endpoint.yaml](endpoint.yaml).

## Структура

| Файл | Что делает |
|------|------------|
| `serve.py` | запуск и остановка `llama-server` (`--slots 1` для опытов 1–2, `--slots 4` для опыта 3), паспорт сервера |
| `run.py` | **один запуск эксперимента** по конфигурации; пишет `runs/<run_id>/` и запись в MLflow |
| `prepare_data.py` | собирает `data/manifest.json`: книга, модель и сервер с источниками, версиями и SHA-256 |
| `make_report.py` | таблицы, графики и `report/REPORT.md` из сохранённых результатов |
| `endpoint.yaml` | адрес сервера, имя модели, параметры генерации по умолчанию |
| `configs/` | `exp1_prompt_length`, `exp1_prefix_cache`, `exp2_output_length`, `exp3_load`, `checks` |
| `lr1/client.py` | потоковый запрос и отметки времени t0, t1, tN (TTFT, TPOT, полное время) |
| `lr1/prompts.py` | запросы заданной длины из «Pride and Prejudice»: метка + инструкция + фрагмент текста |
| `lr1/experiments.py` | четыре опыта и проверки; сводки пересчитываются из `results.jsonl` |
| `lr1/metrics.py` | TTFT, TPOT, процентили, прямая по МНК, goodput, размер KV-кеша на токен |
| `lr1/server.py` | команда запуска, разбор журнала сервера, паспорт |
| `server/` | паспорта сервера (`passport_np1.json`, `passport_np4.json`) |
| `data/` | `pg1342.txt` (текст книги), `manifest.json` |
| `tests/` | проверки реализации на сервере-заглушке |

Каталог запуска `runs/<run_id>/`: `config.yaml`, `metadata.json` (коммит, версии, сервер, загрузка CPU до и после),
`results.jsonl` (запись на каждый запрос, дописывается по мере получения), `metrics.json` (пересчитано из `results.jsonl`),
`snapshots.jsonl` (метрики сервера `/metrics` до, во время и после уровня нагрузки).

## Установка (Windows, PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Модель и сервер в git не лежат, их нужно скачать по ссылкам из `data/manifest.json` и положить так:

| Файл | Куда | Размер | Откуда |
|------|------|--------|--------|
| `llama-b12182-bin-win-cpu-x64.zip` | `tools/` (распаковать в `tools/llama/`) | 24,7 МБ | https://github.com/ggml-org/llama.cpp/releases/download/b12182/llama-b12182-bin-win-cpu-x64.zip |
| `Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf` | `models/` | 2,5 ГБ | https://huggingface.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf |

```powershell
python prepare_data.py      # проверит SHA-256 всех трёх файлов по манифесту
python -m pytest -q         # проверки реализации (сервер-заглушка, около 1–3 минут)
```

## Запуск экспериментов

Замеры задержки чувствительны к нагрузке на процессор: перед итоговым запуском `run.py` измеряет загрузку CPU и отказывается стартовать,
если она выше 25 % (`--allow-busy` отключает проверку, результат тогда нельзя считать чистым). Закройте тяжёлые программы.
Итоговые запуски требуют закоммиченного кода; `--debug` — отладка (`runs_debug/`, эксперимент MLflow `lr1_debug`, в отчёт не входит).

```powershell
# серверный профиль 1: один слот — эксперименты 1 и 2 и проверки
python serve.py start --slots 1
python serve.py passport
python run.py --config configs/checks.yaml
python run.py --config configs/exp1_prompt_length.yaml
python run.py --config configs/exp1_prefix_cache.yaml
python run.py --config configs/exp2_output_length.yaml
python serve.py stop

# серверный профиль 2: четыре слота — эксперимент 3
python serve.py start --slots 4
python serve.py passport
python run.py --config configs/exp3_load.yaml
python serve.py stop

python make_report.py
```

## MLflow

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Открыть http://127.0.0.1:5000, эксперимент `lr1`. Журнал — `mlflow.db`, файлы-артефакты — `mlruns/`; вместе с `runs/`, `server_logs/`
они не попадают в git и сдаются архивом `lr1_artifacts.zip`.
