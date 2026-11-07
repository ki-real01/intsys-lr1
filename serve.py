"""Управление сервером llama-server.

    python serve.py start --slots 1        # эксперименты 1–2: один слот
    python serve.py start --slots 4        # эксперимент 3: четыре слота
    python serve.py status
    python serve.py passport               # паспорт запущенного сервера -> server/passport_np<N>.json
    python serve.py stop
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)

from lr1 import server  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description="ЛР1: llama-server")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start")
    s.add_argument("--slots", type=int, required=True)
    s.add_argument("--ctx", type=int, default=16384)
    s.add_argument("--threads", type=int, default=6)
    s.add_argument("--port", type=int, default=8080)
    sub.add_parser("stop")
    sub.add_parser("status")
    sub.add_parser("passport")
    args = p.parse_args(argv)

    if args.cmd == "start":
        st = server.start(args.slots, args.ctx, args.threads, args.port)
        print("Сервер запущен:", json.dumps(st, ensure_ascii=False, indent=1))
    elif args.cmd == "stop":
        print("Остановлен" if server.stop() else "Сервер не был запущен")
    elif args.cmd == "status":
        if not server.STATE.exists():
            print("не запущен")
        else:
            st = json.loads(server.STATE.read_text(encoding="utf-8"))
            print(f"pid={st['pid']} port={st['port']} slots={st['slots']} ctx={st['ctx']} "
                  f"готов={server.is_healthy(st['port'])}")
    elif args.cmd == "passport":
        pp = server.passport()
        out = Path("server") / f"passport_np{pp['slots']}.json"
        out.write_text(json.dumps(pp, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Записан {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
