"""Сборка и строгая проверка answer.csv (формат из условия)."""
import pandas as pd

from .data import ITEM_ID_RE
from .repro import file_md5


def save_answer(query_ids, predictions, path) -> str:
    answer = pd.DataFrame({
        "query_id": [str(x) for x in query_ids],
        "answer": [" ".join(p) for p in predictions],
    })
    answer.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    return file_md5(path)


def validate_answer(path, bench_query_ids, corpus_item_ids, k: int = 50) -> dict:
    """Все проверки из условия + то, что платформа проглотит молча (чужие/битые item_id)."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    assert list(df.columns) == ["query_id", "answer"], f"колонки: {list(df.columns)}"
    assert df["query_id"].is_unique, "повторы query_id"
    expected = pd.Index([str(x) for x in bench_query_ids])
    assert len(expected) == len(df) and expected.isin(df["query_id"]).all(), "набор query_id не совпадает"
    assert (df["query_id"].str.len() == 16).all(), "query_id должен быть из 16 символов"

    corpus = frozenset(corpus_item_ids)   # только проверка вхождения, без итерации
    sizes = []
    for qid, ans in zip(df["query_id"], df["answer"]):
        ids = ans.split(" ") if ans else []
        assert len(ids) <= k, f"{qid}: больше {k} item_id"
        assert len(ids) == len(dict.fromkeys(ids)), f"{qid}: повторы внутри строки"
        assert all(ITEM_ID_RE.match(x) for x in ids), f"{qid}: item_id не в формате [0-9a-f]{{16}}"
        assert all(x in corpus for x in ids), f"{qid}: есть item_id вне корпуса"
        sizes.append(len(ids))
    return {"rows": len(df), "min_items": min(sizes), "max_items": max(sizes), "md5": file_md5(path)}
