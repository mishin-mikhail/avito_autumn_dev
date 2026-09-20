"""
Смесь двух артефактов эмбеддингов (v8).

Вектора каждой модели нормированы, поэтому склейка [sqrt(w1)·a, sqrt(w2)·b] при w1 + w2 = 1
тоже имеет единичную длину, а скалярное произведение склеенных векторов равно
w1·cos_1 + w2·cos_2. Пайплайн v6 работает с такой смесью без изменений: список кандидатов
по эмбеддингам, признак близости и похожие запросы train считаются по средней близости двух моделей.

Смесь собирается из готовых артефактов (вектора объявлений, запросов и текстов train),
нейросеть не нужна. Результат детерминирован: только перестановка строк, умножение и округление до float16.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import knn_prior
from .repro import file_md5


def _aligned(keys: list, saved_keys: list, matrix: np.ndarray, what: str) -> np.ndarray:
    position = {v: i for i, v in enumerate(saved_keys)}
    missing = [k for k in keys if k not in position]
    if missing:
        raise ValueError(f"в артефакте нет {len(missing):,} {what}, например {missing[:3]}")
    return matrix[np.array([position[k] for k in keys], dtype=np.int64)].astype(np.float32)


def _mix(parts: list, weights: tuple) -> np.ndarray:
    return np.concatenate([np.sqrt(w) * p for p, w in zip(parts, weights)], axis=1).astype(np.float16)


def mix_artifacts(sources: list, weights: tuple, out_dir, overwrite: bool = False) -> dict:
    """sources - папки артефактов, weights - веса моделей (сумма 1). Возвращает манифест смеси."""
    out_dir = Path(out_dir)
    if (out_dir / "manifest.json").is_file() and not overwrite:
        manifest = json.loads((out_dir / "manifest.json").read_text())
        print(f"смесь уже собрана: {out_dir}")
        return manifest
    assert abs(sum(weights) - 1.0) < 1e-9, "сумма весов должна быть 1"
    sources = [Path(s) for s in sources]
    manifests = [json.loads((s / "manifest.json").read_text()) for s in sources]

    # ключи берём из первого артефакта; во всех должны быть те же объявления, запросы и тексты train
    items = pd.read_parquet(sources[0] / "items.parquet")["item_id"].tolist()
    queries = pd.read_parquet(sources[0] / "queries.parquet")["text"].tolist()
    train_texts = pd.read_parquet(sources[0] / knn_prior.TEXTS_FILE)["text"].tolist()
    item_parts, query_parts, train_parts = [], [], []
    for s in sources:
        item_parts.append(_aligned(items, pd.read_parquet(s / "items.parquet")["item_id"].tolist(),
                                   np.load(s / "item_embeddings.npy"), "объявлений"))
        query_parts.append(_aligned(queries, pd.read_parquet(s / "queries.parquet")["text"].tolist(),
                                    np.load(s / "query_embeddings.npy"), "запросов"))
        train_parts.append(_aligned(train_texts, pd.read_parquet(s / knn_prior.TEXTS_FILE)["text"].tolist(),
                                    np.load(s / knn_prior.EMB_FILE), "текстов train"))

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "item_embeddings.npy", _mix(item_parts, weights))
    np.save(out_dir / "query_embeddings.npy", _mix(query_parts, weights))
    pd.DataFrame({"item_id": items}).to_parquet(out_dir / "items.parquet", index=False)
    pd.DataFrame({"text": queries}).to_parquet(out_dir / "queries.parquet", index=False)
    train_emb = _mix(train_parts, weights)
    np.save(out_dir / knn_prior.EMB_FILE, train_emb)
    pd.DataFrame({"text": train_texts}).to_parquet(out_dir / knn_prior.TEXTS_FILE, index=False)

    source_info = [{"dir": s.name, "model_name": m["model_name"], "weight": w, "md5": m["md5"],
                    "recall@100_tuned": m.get("recall@100_tuned")}
                   for s, m, w in zip(sources, manifests, weights)]
    train_manifest = {"sources": source_info, "n_texts": len(train_texts), "dim": int(train_emb.shape[1]),
                      "md5": {f: file_md5(out_dir / f) for f in (knn_prior.TEXTS_FILE, knn_prior.EMB_FILE)}}
    (out_dir / knn_prior.MANIFEST_FILE).write_text(json.dumps(train_manifest, ensure_ascii=False, indent=1))
    files = ["item_embeddings.npy", "query_embeddings.npy", "items.parquet", "queries.parquet"]
    manifest = {"model_name": " + ".join(f"{m['model_name']} ({s.name}, w={w})"
                                         for s, m, w in zip(sources, manifests, weights)),
                "mode": "full" if all(m["mode"] == "full" for m in manifests) else "mixed",
                "recall@100_tuned": float(np.dot(weights, [m.get("recall@100_tuned") or 0.0 for m in manifests])),
                "sources": source_info, "n_items": len(items), "n_queries": len(queries),
                "dim": int(train_emb.shape[1]), "md5": {f: file_md5(out_dir / f) for f in files}}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(f"смесь собрана: {out_dir} | размерность {manifest['dim']} | объявлений {len(items):,}, "
          f"запросов {len(queries):,}, текстов train {len(train_texts):,}")
    return manifest
