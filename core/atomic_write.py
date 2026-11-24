"""Replace a text file only after its complete contents have reached disk."""
from contextlib import contextmanager
import os
import tempfile


@contextmanager
def atomic_text_writer(filepath):
    destination = os.path.abspath(filepath)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=os.path.dirname(destination),
            prefix=".annotation-", suffix=".tmp", delete=False,
        ) as stream:
            temporary = stream.name
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
