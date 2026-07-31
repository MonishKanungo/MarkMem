"""Git backend — the audit trail and versioning layer.

Behind a small Protocol so pygit2/dulwich can slot in later; the default
implementation shells out to the git CLI, which is universally present in the
target environments and keeps the core dependency-free. All history-mutating
callers must hold ``writer_lock`` (single-writer discipline, §5.4).
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol

from filelock import FileLock

from ..obs import log

COMMIT_SEP = "\x1e"
FIELD_SEP = "\x1f"


class GitError(RuntimeError):
    pass


@dataclass
class CommitInfo:
    sha: str
    author: str
    date: str
    message: str
    diff: str = ""


def writer_lock(repo_root: Path, timeout: float = 30.0) -> FileLock:
    lock_dir = repo_root / ".strata" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    return FileLock(str(lock_dir / "writer.lock"), timeout=timeout)


class GitBackend(Protocol):
    def init(self) -> None: ...
    def commit_all(self, message: str) -> Optional[str]: ...
    def rm(self, rel_paths: list[str]) -> None: ...
    def mv(self, rel_from: str, rel_to: str) -> None: ...
    def history(self, rel_path: Optional[str] = None, limit: int = 50,
                include_diff: bool = False) -> list[CommitInfo]: ...
    def gc(self, aggressive: bool = False) -> None: ...


def git_available() -> bool:
    return shutil.which("git") is not None


@dataclass
class SubprocessGit:
    root: Path
    quiet_missing: bool = False
    _identity_checked: bool = field(default=False, repr=False)

    def _run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        proc = subprocess.run(
            ["git", "-C", str(self.root), *args],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if check and proc.returncode != 0:
            raise GitError(f"git {' '.join(args[:2])} failed: {proc.stderr.strip() or proc.stdout.strip()}")
        return proc

    def init(self) -> None:
        if not git_available():
            raise GitError("git executable not found on PATH — Strata requires git (see `strata doctor`)")
        if not (self.root / ".git").exists():
            self._run("init", "-q")
        # Byte-stable storage on Windows; readable non-ascii paths in status output.
        self._run("config", "core.autocrlf", "false")
        self._run("config", "core.quotepath", "false")
        self._ensure_identity()

    def _ensure_identity(self) -> None:
        if self._identity_checked:
            return
        for key, default in (("user.name", "strata"), ("user.email", "strata@localhost")):
            if not self._run("config", key, check=False).stdout.strip():
                self._run("config", "--local", key, default)
        self._identity_checked = True

    def _dirty(self) -> bool:
        return bool(self._run("status", "--porcelain").stdout.strip())

    def commit_all(self, message: str) -> Optional[str]:
        """Stage everything and commit once (debounced batch commit). Returns the
        new commit sha, or None if there was nothing to commit."""
        self._ensure_identity()
        if not self._dirty():
            return None
        self._run("add", "-A")
        self._run("commit", "-q", "-m", message)
        return self._run("rev-parse", "HEAD").stdout.strip()

    def rm(self, rel_paths: list[str]) -> None:
        existing = [p for p in rel_paths if (self.root / p).exists()]
        if existing:
            self._run("rm", "-r", "-q", "--ignore-unmatch", "--", *existing)

    def mv(self, rel_from: str, rel_to: str) -> None:
        (self.root / rel_to).parent.mkdir(parents=True, exist_ok=True)
        self._run("mv", rel_from, rel_to)

    def history(self, rel_path: Optional[str] = None, limit: int = 50,
                include_diff: bool = False) -> list[CommitInfo]:
        fmt = f"%H{FIELD_SEP}%an{FIELD_SEP}%aI{FIELD_SEP}%s{COMMIT_SEP}"
        args = ["log", f"--format={fmt}", f"-{limit}"]
        if rel_path:
            args += ["--follow", "--", rel_path]
        proc = self._run(*args, check=False)
        if proc.returncode != 0:   # empty repo / unborn branch
            return []
        commits: list[CommitInfo] = []
        for chunk in proc.stdout.split(COMMIT_SEP):
            chunk = chunk.strip()
            if not chunk:
                continue
            parts = chunk.split(FIELD_SEP)
            if len(parts) != 4:
                continue
            info = CommitInfo(sha=parts[0], author=parts[1], date=parts[2], message=parts[3])
            if include_diff:
                show_args = ["show", info.sha, "--format="]
                if rel_path:
                    show_args += ["--", rel_path]
                info.diff = self._run(*show_args, check=False).stdout
            commits.append(info)
        return commits

    def gc(self, aggressive: bool = False) -> None:
        args = ["gc", "--quiet", "--prune=now"]
        if aggressive:
            args.insert(1, "--aggressive")
        proc = self._run(*args, check=False)
        if proc.returncode != 0:
            log.warning("git gc failed: %s", proc.stderr.strip())

    def filter_repo_available(self) -> bool:
        return self._run("filter-repo", "--version", check=False).returncode == 0

    def filter_out_paths(self, rel_paths: list[str]) -> None:
        """Rewrite history to remove paths entirely (erasure mode 2, §4.1).
        Requires git-filter-repo. Invalidates clones — documented caveat."""
        if not self.filter_repo_available():
            raise GitError(
                "git-filter-repo is not installed (pip install git-filter-repo). "
                "History-rewrite erasure needs it; `scrub` mode works without."
            )
        args = ["filter-repo", "--force", "--invert-paths"]
        for p in rel_paths:
            args += ["--path", p.replace("\\", "/")]
        self._run(*args)
        self.gc(aggressive=True)
