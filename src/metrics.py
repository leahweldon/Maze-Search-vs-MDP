from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator


@dataclass
class SearchStats:
    nodes_expanded: int = 0
    frontier_max: int = 0


@contextmanager
def timer_ms() -> Iterator[callable]:
    start = time.perf_counter()
    yield lambda: (time.perf_counter() - start) * 1000.0