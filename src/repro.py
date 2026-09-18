"""
Всё, что отвечает за воспроизводимость.

Правила проекта (соблюдаются во всех модулях):
  * никаких итераций по set() строк: порядок зависит от хеш-сида процесса;
    дедупликация только через dict.fromkeys / np.unique / сортировку;
  * любой отбор топ-K идёт по ключу (округлённый скор по убыванию, item_id по возрастанию),
    поэтому «случайных» ничьих нет;
  * разбиения делаются по md5 от ключа, а не через встроенный hash();
  * сэмплирование — только через явный np.random.default_rng(seed).
"""
import hashlib
import os
import random
from pathlib import Path

_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def set_thread_env(n_threads: int) -> None:
    """Фиксирует число потоков BLAS. Действует, только если вызвана ДО импорта numpy."""
    for v in _THREAD_VARS:
        os.environ[v] = str(n_threads)


def seed_everything(seed: int) -> None:
    """Фиксирует все генераторы случайных чисел. Вызывается первой в каждой точке входа."""
    # cuBLAS читает эту переменную при инициализации CUDA, поэтому ставим её заранее
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    # PYTHONHASHSEED влияет только на дочерние процессы; в текущем процессе
    # защищаемся правилом «не итерироваться по set строк» (см. докстринг модуля)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    import numpy as np
    np.random.seed(seed)

    try:  # torch в бейзлайне не нужен, но понадобится на следующих этапах
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    except ImportError:
        pass


def md5_hex(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def md5_unit(s: str) -> float:
    """Детерминированное «случайное» число в [0, 1) по строке."""
    return int(md5_hex(s)[:12], 16) / float(1 << 48)


def file_md5(path) -> str:
    h = hashlib.md5()
    with open(Path(path), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def library_versions() -> dict:
    """Версии ключевых библиотек — печатаем в логе и фиксируем в README."""
    import importlib
    import platform
    out = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "scipy", "sklearn", "pyarrow", "pymorphy3", "lightgbm", "torch"):
        try:
            out[name] = importlib.import_module(name).__version__
        except Exception:  # noqa: BLE001 — модуль может отсутствовать
            out[name] = None
    return out
