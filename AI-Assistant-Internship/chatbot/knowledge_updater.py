"""Periodic knowledge-base expansion from configured sources."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chatbot.knowledge_base import DynamicKnowledgeBase


@dataclass(frozen=True)
class KnowledgeSource:
    """A local file or directory watched for knowledge-base updates."""

    path: Path
    enabled: bool = True


class KnowledgeBaseUpdater:
    """Scan configured sources and ingest new or changed documents."""

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}

    def __init__(
        self,
        knowledge_base: "DynamicKnowledgeBase",
        sources_file: Path,
        state_file: Path,
    ) -> None:
        self.knowledge_base = knowledge_base
        self.sources_file = sources_file
        self.state_file = state_file

    def load_sources(self) -> list[KnowledgeSource]:
        """Load source definitions from JSON."""

        if not self.sources_file.exists():
            return []
        payload = json.loads(self.sources_file.read_text(encoding="utf-8"))
        sources = payload.get("sources", [])
        return [
            KnowledgeSource(path=Path(item["path"]), enabled=bool(item.get("enabled", True)))
            for item in sources
            if item.get("path")
        ]

    def update(self) -> dict[str, object]:
        """Ingest changed files and persist update state."""

        state = self._load_state()
        changed_files = self._changed_files(self.load_sources(), state)
        chunks_added = 0
        ingested: list[str] = []
        for path in changed_files:
            try:
                chunks_added += self.knowledge_base.add_file(path)
                state[str(path.resolve())] = self._fingerprint(path)
                ingested.append(str(path))
            except Exception as exc:  # pragma: no cover - file dependent
                state[f"error:{path.resolve()}"] = str(exc)

        state["_last_run_utc"] = datetime.now(timezone.utc).isoformat()
        state["_last_files_ingested"] = ingested
        state["_last_chunks_added"] = chunks_added
        self._save_state(state)
        return {"files_ingested": ingested, "chunks_added": chunks_added, "last_run_utc": state["_last_run_utc"]}

    def run_periodically(
        self,
        interval_minutes: int = 60,
        stop_after_runs: int | None = None,
        on_result: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        """Run the incremental updater on a fixed interval.

        This method is intentionally simple so it works in local scripts, CI, or a
        long-running server process. Production deployments can call the same
        `update` method from cron, Windows Task Scheduler, GitHub Actions, or a
        container scheduler.
        """

        runs = 0
        while stop_after_runs is None or runs < stop_after_runs:
            result = self.update()
            if on_result:
                on_result(result)
            runs += 1
            if stop_after_runs is not None and runs >= stop_after_runs:
                break
            time.sleep(max(interval_minutes, 1) * 60)

    def _changed_files(self, sources: list[KnowledgeSource], state: dict[str, object]) -> list[Path]:
        """Return supported files that are new or changed."""

        changed: list[Path] = []
        for source in sources:
            if not source.enabled:
                continue
            path = source.path.expanduser()
            candidates = [path] if path.is_file() else list(path.rglob("*")) if path.is_dir() else []
            for candidate in candidates:
                if candidate.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                    continue
                key = str(candidate.resolve())
                fingerprint = self._fingerprint(candidate)
                if state.get(key) != fingerprint:
                    changed.append(candidate)
        return changed

    @staticmethod
    def _fingerprint(path: Path) -> dict[str, object]:
        """Return a stable incremental-index fingerprint for a source file."""

        stat = path.stat()
        digest = sha256(path.read_bytes()).hexdigest()
        return {"sha256": digest, "size": stat.st_size, "modified": stat.st_mtime}

    def _load_state(self) -> dict[str, object]:
        """Load update state."""

        if not self.state_file.exists():
            return {}
        return json.loads(self.state_file.read_text(encoding="utf-8"))

    def _save_state(self, state: dict[str, object]) -> None:
        """Persist update state."""

        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")
