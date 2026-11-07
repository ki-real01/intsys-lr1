"""Запросы заданной длины из «Pride and Prejudice» (Project Gutenberg № 1342).

Запрос: уникальная метка в самом начале + инструкция + фрагмент текста нужной длины в токенах.
Метка в начале не даёт кешированию префикса переиспользовать прошлые запросы.
Токены считает токенизатор сервера (/tokenize, /detokenize), длину итогового запроса после
шаблона диалога проверяет usage.prompt_tokens.
"""
import hashlib
import re
from pathlib import Path

import httpx


def load_book(path):
    """(текст книги без заголовка и лицензии Project Gutenberg, SHA-256 файла).

    Хеш считается по тем же байтам, из которых читается текст."""
    raw = Path(path).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    text = raw.decode("utf-8-sig").replace("\r\n", "\n")
    m_start = re.search(r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK[^\n]*\*\*\*\n", text)
    m_end = re.search(r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK[^\n]*\*\*\*", text)
    if not m_start or not m_end or m_end.start() <= m_start.end():
        raise ValueError(f"В {path} не найдены маркеры начала и конца книги Project Gutenberg")
    body = text[m_start.end():m_end.start()].strip()
    if len(body) < 100_000:
        raise ValueError("Текст книги слишком короткий")
    return body, sha


class ServerTokenizer:
    """Токенизатор модели через HTTP-интерфейс llama-server."""

    def __init__(self, server_root, timeout=120.0):
        self.http = httpx.Client(base_url=server_root, timeout=timeout)

    def tokenize(self, text):
        r = self.http.post("/tokenize", json={"content": text, "add_special": False})
        r.raise_for_status()
        return r.json()["tokens"]

    def detokenize(self, tokens):
        r = self.http.post("/detokenize", json={"tokens": tokens})
        r.raise_for_status()
        return r.json()["content"]


class PromptBuilder:
    def __init__(self, tokenizer, book_text, instruction):
        self.tokenizer = tokenizer
        self.instruction = instruction
        self.book_tokens = tokenizer.tokenize(book_text)  # один раз на весь запуск
        self.overhead = None  # токены метки, инструкции и шаблона диалога

    @staticmethod
    def label(index):
        return f"req-{index:05d}"  # фиксированная ширина: накладные токены одинаковы

    def passage(self, n_tokens, offset=0):
        if n_tokens < 0 or offset < 0 or offset + n_tokens > len(self.book_tokens):
            raise ValueError(f"Фрагмент {n_tokens} токенов с позиции {offset} выходит за пределы книги "
                             f"({len(self.book_tokens)} токенов)")
        return self.tokenizer.detokenize(self.book_tokens[offset:offset + n_tokens]) if n_tokens else ""

    def messages(self, index, n_text_tokens, offset=0):
        body = f"[{self.label(index)}] {self.instruction}\n\n{self.passage(n_text_tokens, offset)}"
        return [{"role": "user", "content": body}]

    def calibrate(self, probe_prompt_tokens, probe_text_tokens):
        """Накладные токены = usage.prompt_tokens пробного запроса - длина его текста."""
        self.overhead = probe_prompt_tokens - probe_text_tokens
        return self.overhead

    def text_tokens_for(self, target_prompt_tokens):
        if self.overhead is None:
            raise RuntimeError("Сначала calibrate()")
        n = target_prompt_tokens - self.overhead
        if n < 1:
            raise ValueError(f"Длина {target_prompt_tokens} меньше накладных токенов ({self.overhead})")
        return n

    def build(self, index, target_prompt_tokens, offset=0):
        """messages для запроса длиной target_prompt_tokens токенов после шаблона диалога."""
        return self.messages(index, self.text_tokens_for(target_prompt_tokens), offset)
