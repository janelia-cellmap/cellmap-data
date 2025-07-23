import threading
import queue
from typing import Callable, Any, Optional


class Prefetcher:
    """
    Asynchronous prefetcher for CellMapDataset and DataLoader.
    Uses a background thread to load data ahead of time into a queue.
    Supports custom fetch functions and batch sizes.
    """

    def __init__(self, fetch_fn: Callable[[Any], Any], max_prefetch: int = 8):
        self.fetch_fn = fetch_fn
        self.max_prefetch = max_prefetch
        self._queue = queue.Queue(max_prefetch)
        self._thread = None
        self._stop_event = threading.Event()

    def start(self, indices):
        def _worker():
            for idx in indices:
                if self._stop_event.is_set():
                    break
                item = self.fetch_fn(idx)
                self._queue.put(item)

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()

    def get(self, timeout: Optional[float] = None):
        return self._queue.get(timeout=timeout)

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join()


# Usage in CellMapDataset/CellMapDataLoader:
#   prefetcher = Prefetcher(self.__getitem__, max_prefetch=batch_size*2)
#   prefetcher.start(indices)
#   for _ in range(batch_size):
#       batch = prefetcher.get()
#   prefetcher.stop()
