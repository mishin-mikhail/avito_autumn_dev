"""Мелкие вспомогательные функции: таймер, заголовки в логах, отчёт о ресурсах машины."""
import os
import shutil
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def timer(name: str):
    """Печатает время выполнения блока: `with timer("этап"): ...`."""
    start = time.perf_counter()
    yield
    print(f"[{name}] {time.perf_counter() - start:.1f} c")


def section(title: str) -> None:
    """Заголовок блока в текстовом выводе ячейки."""
    print(f"\n── {title} " + "─" * max(0, 70 - len(title)))


def _read_int(path: str):
    try:
        value = Path(path).read_text().strip()
        return None if value in ("", "max") else int(value)
    except (OSError, ValueError):
        return None


def memory_gb() -> dict:
    """Оперативная память с учётом лимита контейнера (cgroup): в онлайн-средах `free`
    часто показывает память всего сервера, а доступна только выделенная часть."""
    total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    available = total
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                available = int(line.split()[1]) * 1024
    except OSError:
        pass
    limit = _read_int("/sys/fs/cgroup/memory.max") or _read_int("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if limit and limit < total:
        used = (_read_int("/sys/fs/cgroup/memory.current")
                or _read_int("/sys/fs/cgroup/memory/memory.usage_in_bytes") or 0)
        total, available = limit, max(limit - used, 0)
    return {"total": round(total / 2 ** 30, 1), "available": round(available / 2 ** 30, 1)}


def resources_report(work_dir, need_ram_gb: float, need_disk_gb: float) -> None:
    """Печатает RAM, диск и число ядер; предупреждает, если ресурсов может не хватить."""
    mem = memory_gb()
    disk = shutil.disk_usage(work_dir).free / 2 ** 30
    print(f"RAM: {mem['available']} ГБ свободно из {mem['total']} | диск в {work_dir}: {disk:.0f} ГБ свободно | "
          f"ядер CPU: {os.cpu_count()}")
    if mem["available"] < need_ram_gb:
        print(f"[warn] ноутбуку нужно около {need_ram_gb} ГБ RAM — возможна нехватка памяти")
    if disk < need_disk_gb:
        print(f"[warn] ноутбуку нужно около {need_disk_gb} ГБ на диске")
