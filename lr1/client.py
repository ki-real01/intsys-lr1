"""Клиент-измеритель: потоковый запрос к OpenAI-совместимому серверу и отметки времени.

t0 — перед отправкой, t1 — первый фрагмент с текстом, tN — последний фрагмент с текстом.
Число токенов берётся из поля usage, а не из числа фрагментов: сервер может прислать
несколько токенов в одном фрагменте.
"""
import time

from . import metrics


class StreamTimer:
    """Накапливает фрагменты потока и считает метрики. Не зависит от сети."""

    def __init__(self, t0):
        self.t0 = t0
        self.t1 = None
        self.t_last = None
        self.n_chunks = 0
        self.text = []
        self.usage = None
        self.finish_reason = None
        self.timings = None

    def feed(self, chunk, now):
        extra = getattr(chunk, "model_extra", None) or {}
        if extra.get("timings") is not None:
            self.timings = extra["timings"]
        if getattr(chunk, "usage", None) is not None:
            self.usage = chunk.usage  # последний фрагмент потока — только usage
        choices = getattr(chunk, "choices", None)
        if choices:
            choice = choices[0]
            if getattr(choice, "finish_reason", None):
                self.finish_reason = choice.finish_reason
            content = getattr(getattr(choice, "delta", None), "content", None)
            if content:
                if self.t1 is None:
                    self.t1 = now
                self.t_last = now
                self.n_chunks += 1
                self.text.append(content)

    def result(self, error=None):
        n_out = getattr(self.usage, "completion_tokens", None)
        n_in = getattr(self.usage, "prompt_tokens", None)
        rec = {
            "prompt_tokens": n_in, "completion_tokens": n_out,
            "ttft_s": None, "tpot_s": None, "e2e_s": None,
            "finish_reason": self.finish_reason, "n_chunks": self.n_chunks,
            "text": "".join(self.text), "timings": self.timings,
            "t0_s": self.t0, "t_end_s": self.t_last,
        }
        if error is None and (self.t1 is None or self.usage is None):
            error = "ответ без токенов или без usage"
        if error is not None:
            rec.update(status="error", error=error)
            return rec
        rec.update(
            status="ok", error=None,
            ttft_s=metrics.ttft(self.t0, self.t1),
            tpot_s=metrics.tpot(self.t1, self.t_last, n_out),
            e2e_s=metrics.e2e(self.t0, self.t_last),
        )
        return rec


def _call_args(cfg, req):
    body = {"ignore_eos": cfg["request"]["ignore_eos"], "cache_prompt": cfg["request"]["cache_prompt"]}
    body.update(req.get("extra_body") or {})
    return dict(
        model=cfg["endpoint"]["model"], messages=req["messages"], max_tokens=req["max_tokens"],
        temperature=cfg["generation"]["temperature"],  # температура всегда передаётся явно
        stream=True, stream_options={"include_usage": True}, extra_body=body,
    )


def _finish(rec, req):
    rec.update(id=req["id"], condition=req["condition"], warmup=req.get("warmup", False),
               max_tokens=req["max_tokens"])
    return rec


def run_request(client, cfg, req, clock=time.perf_counter):
    """Один потоковый запрос. Ошибка не прерывает серию: она записывается в результат."""
    t0 = clock()
    timer = StreamTimer(t0)
    try:
        stream = client.chat.completions.create(**_call_args(cfg, req))
        for chunk in stream:
            timer.feed(chunk, clock())
        rec = timer.result()
    except Exception as exc:  # noqa: BLE001 — отказ запроса сохраняем, а не теряем
        rec = timer.result(error=f"{type(exc).__name__}: {exc}")
    return _finish(rec, req)


async def arun_request(client, cfg, req, clock=time.perf_counter):
    """То же для AsyncOpenAI (нагрузочный опыт)."""
    t0 = clock()
    timer = StreamTimer(t0)
    try:
        stream = await client.chat.completions.create(**_call_args(cfg, req))
        async for chunk in stream:
            timer.feed(chunk, clock())
        rec = timer.result()
    except Exception as exc:  # noqa: BLE001
        rec = timer.result(error=f"{type(exc).__name__}: {exc}")
    return _finish(rec, req)
