from __future__ import annotations


def log_title(message: str) -> None:
    total_length = 80
    padding = max(0, total_length - len(message) - 2)
    left = "=" * (padding // 2)
    right = "=" * (padding - len(left))
    print(f"{left} {message} {right}")
