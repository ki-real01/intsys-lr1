"""Тестовый сервер-заглушка с OpenAI-совместимым потоковым API и интерфейсом llama-server.

Нужен только для проверки клиента и каркаса без настоящей модели. Задержки задаются явно:
prefill — prefill_s_per_token на токен запроса, decode — token_s на токен ответа.
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeLLM:
    def __init__(self, prefill_s_per_token=0.0005, token_s=0.01, slots=1):
        self.prefill_s_per_token = prefill_s_per_token
        self.token_s = token_s
        self.slots = slots
        self.requests = []
        self.lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # тишина в выводе тестов
                pass

            def _json(self, obj, code=200):
                data = json.dumps(obj).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/health":
                    self._json({"status": "ok"})
                elif self.path == "/props":
                    self._json({"total_slots": outer.slots, "model_path": "fake.gguf", "build_info": "fake"})
                elif self.path == "/metrics":
                    text = ("llamacpp:requests_processing 0\nllamacpp:requests_deferred 0\n"
                            "llamacpp:kv_cache_usage_ratio 0.1\n")
                    data = text.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    self._json({"error": "not found"}, 404)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])).decode("utf-8"))
                if self.path == "/tokenize":
                    self._json({"tokens": [ord(c) for c in body["content"]]})
                elif self.path == "/detokenize":
                    self._json({"content": "".join(chr(t) for t in body["tokens"])})
                elif self.path == "/v1/chat/completions":
                    self._chat(body)
                else:
                    self._json({"error": "not found"}, 404)

            def _chat(self, body):
                with outer.lock:
                    outer.requests.append(body)
                n_in = sum(len(m["content"]) for m in body["messages"]) + 4
                n_out = int(body["max_tokens"]) if body.get("ignore_eos") else min(int(body["max_tokens"]), 5)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()

                def send(obj):
                    self.wfile.write(b"data: " + json.dumps(obj).encode("utf-8") + b"\n\n")
                    self.wfile.flush()

                base = {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "fake"}
                send({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": None},
                                           "finish_reason": None}]})
                time.sleep(n_in * outer.prefill_s_per_token)
                for i in range(n_out):
                    send({**base, "choices": [{"index": 0, "delta": {"content": f"t{i} "}, "finish_reason": None}]})
                    time.sleep(outer.token_s)
                send({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]})
                send({**base, "choices": [], "usage": {"prompt_tokens": n_in, "completion_tokens": n_out,
                                                       "total_tokens": n_in + n_out},
                      "timings": {"cache_n": 0, "prompt_n": n_in}})
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()

    @property
    def root(self):
        return f"http://127.0.0.1:{self.port}"
