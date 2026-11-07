"""Собирает data/manifest.json: текст книги, модель и сервер с источниками, версиями и SHA-256.

    python prepare_data.py

Файлы (книга, модель, архив llama.cpp) скачиваются вручную по ссылкам из манифеста; скрипт
только считает хеши и записывает их. Если манифест уже есть, хеши не меняются молча:
при расхождении скрипт останавливается.
"""
import json
import os
import re
import sys
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)

from lr1.prompts import load_book  # noqa: E402
from lr1.tracking import sha256_file  # noqa: E402

BOOK = Path("data/pg1342.txt")
MODEL = Path("models/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
LLAMA_ZIP = Path("tools/llama-b12182-bin-win-cpu-x64.zip")
MANIFEST = Path("data/manifest.json")


def build():
    body, book_sha = load_book(BOOK)
    raw = BOOK.read_bytes()
    gutenberg_id = re.search(rb"\[eBook #(\d+)\]", raw)
    return {
        "datasets": {
            "pride_and_prejudice": {
                "source": "Project Gutenberg, Austen J. Pride and Prejudice, eBook № 1342",
                "url": "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
                "downloaded": "2026-11-07",
                "license": "общественное достояние (Project Gutenberg License)",
                "selection_rule": ("берётся текст между строками «*** START OF THE PROJECT GUTENBERG EBOOK» и "
                                   "«*** END OF THE PROJECT GUTENBERG EBOOK», переводы строк CRLF заменяются на LF"),
                "records_total_chars_in_file": len(raw),
                "records_used_chars": len(body),
                "excluded": "заголовок и лицензия Project Gutenberg",
                "gutenberg_ebook_id": int(gutenberg_id.group(1)) if gutenberg_id else 1342,
                "files": {"data/pg1342.txt": {"sha256": book_sha, "bytes": len(raw)}},
            }
        },
        "models": {
            "qwen3_4b_q4km": {
                "base_model": "Qwen/Qwen3-4B-Instruct-2507",
                "base_model_revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
                "base_model_config": {"num_hidden_layers": 36, "num_attention_heads": 32, "num_key_value_heads": 8,
                                      "head_dim": 128, "hidden_size": 2560, "max_position_embeddings": 262144,
                                      "torch_dtype": "bfloat16",
                                      "source": "https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/raw/main/config.json"},
                "gguf_repo": "bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF",
                "gguf_revision": "ae44f08e1392f39c0e474af10c3ff8355c8b6688",
                "gguf_url": "https://huggingface.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF/resolve/main/"
                            "Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
                "file": MODEL.name, "format": "GGUF, квантование Q4_K_M",
                "bytes": MODEL.stat().st_size, "sha256": sha256_file(MODEL),
                "note": "официальных GGUF у модели нет; взята популярная сборка сообщества (bartowski)",
            }
        },
        "tools": {
            "llama_cpp": {"version": "b12182 (0.9.0-dev, commit fd27b2034)",
                          "url": "https://github.com/ggml-org/llama.cpp/releases/download/b12182/"
                                 "llama-b12182-bin-win-cpu-x64.zip",
                          "file": LLAMA_ZIP.name, "bytes": LLAMA_ZIP.stat().st_size, "zip_sha256": sha256_file(LLAMA_ZIP)}
        },
    }


def main():
    new = build()
    if MANIFEST.exists():
        old = json.loads(MANIFEST.read_text(encoding="utf-8"))
        pairs = [(old["datasets"]["pride_and_prejudice"]["files"]["data/pg1342.txt"]["sha256"],
                  new["datasets"]["pride_and_prejudice"]["files"]["data/pg1342.txt"]["sha256"], "книга"),
                 (old["models"]["qwen3_4b_q4km"]["sha256"], new["models"]["qwen3_4b_q4km"]["sha256"], "модель"),
                 (old["tools"]["llama_cpp"]["zip_sha256"], new["tools"]["llama_cpp"]["zip_sha256"], "llama.cpp")]
        bad = [name for a, b, name in pairs if a != b]
        if bad:
            sys.exit(f"SHA-256 расходится с манифестом: {', '.join(bad)}. Файлы изменились или скачаны другие.")
        print("data/manifest.json уже есть, хеши совпадают")
        return
    MANIFEST.write_text(json.dumps(new, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Записан data/manifest.json")


if __name__ == "__main__":
    main()
