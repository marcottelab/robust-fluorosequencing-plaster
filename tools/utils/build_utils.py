from pathlib import Path


def out_of_date(source: str, target: str):
    """Return True if source's modification time is more recent than target's"""

    source_path = Path(source)
    target_path = Path(target)

    source_time = source_path.stat().st_mtime
    if target_path.exists():
        target_time = target_path.stat().st_mtime
        return source_time > target_time
    else:
        return True
