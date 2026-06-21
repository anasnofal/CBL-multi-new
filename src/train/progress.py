import json
from datetime import datetime, timezone
from pathlib import Path


class ProgressTracker:
    """
    Tracks which LSOAs have been successfully trained, persisted as a JSON file.

    Format::

        {
          "E01012480": "2025-01-01T12:00:00+00:00",
          "E01017991": "2025-01-01T12:05:00+00:00"
        }

    Each key is an LSOA code; the value is the UTC timestamp of completion.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._done: dict[str, str] = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._done, indent=2), encoding="utf-8")

    def is_done(self, lsoa_code: str) -> bool:
        return lsoa_code in self._done

    def mark_done(self, lsoa_codes) -> None:
        """Record one or more LSOA codes as completed (single disk write)."""
        ts = datetime.now(timezone.utc).isoformat()
        for code in ([lsoa_codes] if isinstance(lsoa_codes, str) else lsoa_codes):
            self._done[code] = ts
        self._save()

    def reset(self) -> None:
        """Clear all progress and delete the JSON file."""
        self._done = {}
        if self.path.exists():
            self.path.unlink()

    @property
    def trained(self) -> set[str]:
        return set(self._done.keys())

    def __len__(self) -> int:
        return len(self._done)
