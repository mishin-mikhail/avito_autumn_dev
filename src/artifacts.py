"""
Поиск артефакта эмбеддингов (v4).

Ноутбук 03 обучает энкодер и сохраняет в одну папку модель (model/), вектора объявлений
и вектора запросов. Ноутбук 04 ищет эту папку по порядку:
  1. переменная окружения EMB_DIR (или настройка EMB_DIR в первой ячейке ноутбука);
  2. <папка артефактов>/embeddings - туда её кладёт ноутбук 03 на той же машине;
  3. на Kaggle - любая подключённая папка внутри /kaggle/input.
"""
import os
from pathlib import Path

from .paths import KAGGLE_INPUT, get_work_dir, is_kaggle

MARKER = "item_embeddings.npy"


def find_embeddings_dir(required: bool = True, name: str = "embeddings"):
    """name - папка артефакта внутри WORK_DIR (v6: embeddings_v6)."""
    env = os.environ.get("EMB_DIR")
    candidates = [Path(env)] if env else []
    candidates.append(get_work_dir() / name)
    if is_kaggle():
        candidates += sorted({p.parent for p in KAGGLE_INPUT.glob(f"**/{MARKER}")})

    for directory in candidates:
        if (directory / MARKER).is_file() and (directory / "manifest.json").is_file():
            return directory
    if required:
        raise FileNotFoundError(
            f"Не найден артефакт эмбеддингов ({MARKER} + manifest.json). Сначала выполните "
            "03_embeddings.ipynb или укажите путь к папке артефакта в EMB_DIR. "
            f"Проверено: {[str(c) for c in candidates]}")
    return None
