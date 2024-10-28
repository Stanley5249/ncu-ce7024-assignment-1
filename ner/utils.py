from itertools import groupby
from os import PathLike
from pathlib import Path
from typing import Any


def _split_word_and_label(token_tag: str) -> list[str]:
    try:
        return token_tag.rsplit("\t", 1)
    except ValueError:
        raise ValueError(
            f"expected 2 tab-separated fields, got {token_tag!r}"
        ) from None


def _parse_word_and_labels(s: str) -> dict[str, list[list[Any]]]:
    col_words = []
    col_labels = []
    for k, g in groupby(s.splitlines(), bool):
        if k:
            words = []
            labels = []
            for word, label in map(_split_word_and_label, g):
                words.append(word)
                labels.append(label)
            col_words.append(words)
            col_labels.append(labels)
    return {
        "words": col_words,
        "labels": col_labels,
    }


def _parse_words(data: str) -> dict[str, list[list[Any]]]:
    return {"words": [[*g] for k, g in groupby(data.splitlines(), bool) if k]}


def load_wnut16_dataset(
    path: str | PathLike[str], *, with_labels: bool
) -> dict[str, list[list[Any]]]:
    text = Path(path).read_text("utf-8")
    if with_labels:
        return _parse_word_and_labels(text)
    return _parse_words(text)


def align_words[T](tokens: list[T], indices: list[int | None]) -> list[T]:
    res = []
    pre = None
    for cur in indices:
        res.append(tokens[cur] if cur is not None and cur != pre else -100)
        pre = cur
    return res


def unalign_tokens[T](tokens: list[T], indices: list[int | None]) -> list[T]:
    res = []
    pre = None
    for token_id, cur in zip(tokens, indices):
        if cur is None or pre == cur:
            continue
        pre = cur
        res.append(token_id)
    return res
