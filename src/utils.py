"""Мелкие вспомогательные функции для логов."""
import time
from contextlib import contextmanager


@contextmanager
def timer(name: str):
    """Печатает время выполнения блока: `with timer("этап"): ...`."""
    start = time.perf_counter()
    yield
    print(f"[{name}] {time.perf_counter() - start:.1f} c")


def section(title: str) -> None:
    """Заголовок блока в текстовом выводе ячейки."""
    print(f"\n── {title} " + "─" * max(0, 70 - len(title)))
