"""
Пути к данным и артефактам с автоопределением среды (Kaggle / локально).

Порядок поиска данных:
  1. переменная окружения DATA_DIR;
  2. на Kaggle - первая (в отсортированном порядке) папка внутри /kaggle/input,
     где лежат все три parquet-файла (имя датасета в код не зашито);
  3. локально - <корень репозитория>/data.
"""
import os
from pathlib import Path

REQUIRED_FILES = ("train.parquet", "benchmark_queries.parquet", "benchmark_items.parquet")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")


def is_kaggle() -> bool:
    return KAGGLE_INPUT.exists()


def _has_all_files(d: Path) -> bool:
    return all((d / f).is_file() for f in REQUIRED_FILES)


def _find_dirs_with(root: Path, filename: str, max_depth: int = 6) -> list:
    """Папки внутри root, где есть filename.
    os.walk(followlinks=True): Kaggle монтирует датасеты через симлинки, а Path.rglob
    в Python < 3.13 в симлинки на папки не заходит. Результат сортируется - выбор детерминирован."""
    found = []
    root_depth = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        if len(Path(dirpath).parts) - root_depth >= max_depth:
            dirnames[:] = []
        dirnames.sort()
        if filename in filenames:
            found.append(Path(dirpath))
    return sorted(found)


def _describe_tree(root: Path, max_depth: int = 4, limit: int = 40) -> str:
    """Короткий листинг для текста ошибки: что реально лежит в /kaggle/input."""
    lines = []
    root_depth = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        depth = len(Path(dirpath).parts) - root_depth
        if depth >= max_depth:
            dirnames[:] = []
        dirnames.sort()
        lines.append(f"{dirpath}/  файлы: {sorted(filenames)[:5]}")
        if len(lines) >= limit:
            break
    return "\n".join(lines) or "(пусто)"


def get_data_dir() -> Path:
    env = os.environ.get("DATA_DIR")
    if env:
        candidates = [Path(env)]
    elif is_kaggle():
        candidates = _find_dirs_with(KAGGLE_INPUT, "benchmark_items.parquet")
    else:
        candidates = [PROJECT_ROOT / "data"]

    for d in candidates:
        if _has_all_files(d):
            return d
    hint = f"\nСодержимое {KAGGLE_INPUT}:\n{_describe_tree(KAGGLE_INPUT)}" if is_kaggle() else ""
    raise FileNotFoundError(
        f"Не найдены файлы {REQUIRED_FILES}. Проверено: {[str(c) for c in candidates]}. "
        "Положите их в ./data или укажите путь в переменной окружения DATA_DIR." + hint
    )


def get_work_dir() -> Path:
    """Папка для промежуточных артефактов (пулы, веса, логи)."""
    env = os.environ.get("WORK_DIR")
    if env:
        d = Path(env)
    elif is_kaggle():
        d = KAGGLE_WORKING / "artifacts"
    else:
        d = PROJECT_ROOT / "artifacts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_output_dir() -> Path:
    """Куда кладётся answer.csv (на Kaggle - прямо в /kaggle/working для скачивания)."""
    env = os.environ.get("OUTPUT_DIR")
    if env:
        d = Path(env)
    elif is_kaggle():
        d = KAGGLE_WORKING
    else:
        d = PROJECT_ROOT / "outputs"
    d.mkdir(parents=True, exist_ok=True)
    return d
