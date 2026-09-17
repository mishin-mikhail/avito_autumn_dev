"""
Пути к данным и артефактам с автоопределением среды (Kaggle / локально).

Порядок поиска данных:
  1. переменная окружения DATA_DIR;
  2. на Kaggle — первая (в отсортированном порядке) папка внутри /kaggle/input,
     где лежат все три parquet-файла (имя датасета в код не зашито);
  3. локально — <корень репозитория>/data.
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


def get_data_dir() -> Path:
    env = os.environ.get("DATA_DIR")
    if env:
        candidates = [Path(env)]
    elif is_kaggle():
        # sorted — чтобы при нескольких совпадениях выбор был детерминированным
        candidates = sorted({p.parent for p in KAGGLE_INPUT.rglob("benchmark_items.parquet")})
    else:
        candidates = [PROJECT_ROOT / "data"]

    for d in candidates:
        if _has_all_files(d):
            return d
    raise FileNotFoundError(
        f"Не найдены файлы {REQUIRED_FILES}. Проверено: {[str(c) for c in candidates]}. "
        "Положите их в ./data или укажите путь в переменной окружения DATA_DIR."
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
    """Куда кладётся answer.csv (на Kaggle — прямо в /kaggle/working для скачивания)."""
    env = os.environ.get("OUTPUT_DIR")
    if env:
        d = Path(env)
    elif is_kaggle():
        d = KAGGLE_WORKING
    else:
        d = PROJECT_ROOT / "outputs"
    d.mkdir(parents=True, exist_ok=True)
    return d
