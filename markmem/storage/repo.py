"""Repo — filesystem layout, page/raw IO, index.md generation.

Layout (shard-per-user: every user's data lives under exactly two path
prefixes, which is what makes path-based erasure provable — §4.1):

    <root>/
    ├── schema.md, config.yaml, index.md
    ├── wiki/
    │   ├── g/<type>/<slug>.md              # shared / global pages
    │   └── u/<user>/<type>/<slug>.md       # user-scoped pages
    ├── raw/
    │   ├── g/<source_type>/...             # immutable, append-only
    │   ├── u/<user>/<source_type>/...
    │   └── failed/                         # dead-letter queue (§5.7)
    └── .markmem/                            # DERIVED — delete + `markmem reindex` rebuilds

Repo is pure filesystem; git operations are composed on top by callers.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Iterator, Optional

import frontmatter
import yaml

from ..models import Page, RawEntry
from ..obs import log
from ..util import slugify, utcnow_iso

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"

_TS_SAFE = re.compile(r"[:+]")


def _yaml_dump(meta: dict) -> str:
    return yaml.safe_dump(meta, sort_keys=False, allow_unicode=True, width=100)


def dump_page_text(page: Page, body: str) -> str:
    return f"---\n{_yaml_dump(page.to_frontmatter())}---\n\n{body.strip()}\n"


class RepoError(RuntimeError):
    pass


class Repo:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.wiki_dir = self.root / "wiki"
        self.raw_dir = self.root / "raw"
        self.markmem_dir = self.root / ".markmem"
        self.schema_path = self.root / "schema.md"
        self.config_path = self.root / "config.yaml"
        self.index_path = self.root / "index.md"

    @property
    def is_initialized(self) -> bool:
        return self.schema_path.exists() and self.wiki_dir.exists()

    @classmethod
    def scaffold(cls, root: str | Path) -> "Repo":
        repo = cls(root)
        repo.root.mkdir(parents=True, exist_ok=True)
        for d in (repo.wiki_dir, repo.raw_dir / "failed", repo.markmem_dir):
            d.mkdir(parents=True, exist_ok=True)
        for name in ("schema.md", "config.yaml"):
            target = repo.root / name
            if not target.exists():
                shutil.copyfile(TEMPLATES_DIR / name, target)
        gitignore = repo.root / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text(".markmem/\n", encoding="utf-8")
        if not repo.index_path.exists():
            repo.regenerate_index()
        return repo

    # ---------------- page ids & paths ----------------

    @staticmethod
    def make_page_id(page_type: str, title_or_slug: str, user_id: Optional[str] = None) -> str:
        slug = slugify(title_or_slug)
        type_slug = slugify(page_type, fallback="session")
        if user_id:
            return f"u/{slugify(user_id, fallback='user')}/{type_slug}/{slug}"
        return f"g/{type_slug}/{slug}"

    @staticmethod
    def user_prefix(user_id: str) -> str:
        return f"u/{slugify(user_id, fallback='user')}"

    def user_paths(self, user_id: str) -> list[str]:
        """The (exactly two) repo-relative path prefixes holding a user's data."""
        prefix = self.user_prefix(user_id)
        return [f"wiki/{prefix}", f"raw/{prefix}"]

    def page_path(self, page_id: str) -> Path:
        # Colons are invalid characters in Windows filenames. We replace them
        # with dashes to ensure filesystem compatibility.
        safe_id = page_id.replace(":", "-")
        path = (self.wiki_dir / f"{safe_id}.md").resolve()
        if self.wiki_dir.resolve() not in path.parents:
            raise RepoError(f"page id escapes wiki/: {page_id!r}")
        return path

    def rel(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    # ---------------- pages ----------------

    def write_page(self, page: Page, body: str) -> Path:
        page.updated = page.updated or utcnow_iso()
        path = self.page_path(page.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dump_page_text(page, body), encoding="utf-8")
        return path

    def read_page(self, page_id: str) -> Optional[tuple[Page, str]]:
        path = self.page_path(page_id)
        if not path.exists():
            return None
        return self._parse_page_file(path)

    def page_exists(self, page_id: str) -> bool:
        return self.page_path(page_id).exists()

    def delete_page_file(self, page_id: str) -> bool:
        path = self.page_path(page_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def _parse_page_file(self, path: Path) -> Optional[tuple[Page, str]]:
        try:
            post = frontmatter.loads(path.read_text(encoding="utf-8"))
        except Exception as e:                       # malformed hand-edit: skip, don't crash
            log.warning("unparseable page %s: %s", path, e)
            return None
        meta = dict(post.metadata)
        derived_id = path.relative_to(self.wiki_dir).with_suffix("").as_posix()
        meta["id"] = derived_id                      # the path is authoritative for identity
        parts = derived_id.split("/")
        meta.setdefault("type", parts[2] if parts[0] == "u" and len(parts) >= 4 else
                        (parts[1] if len(parts) >= 3 else "session"))
        meta.setdefault("title", path.stem)
        now = utcnow_iso()
        meta.setdefault("created", now)
        meta.setdefault("updated", meta["created"])
        try:
            return Page.model_validate(meta), post.content
        except Exception as e:
            log.warning("invalid page frontmatter %s: %s", path, e)
            return None

    def iter_pages(self) -> Iterator[tuple[Page, str]]:
        if not self.wiki_dir.exists():
            return
        for path in sorted(self.wiki_dir.rglob("*.md")):
            parsed = self._parse_page_file(path)
            if parsed:
                yield parsed

    def list_pages(self, type: Optional[str] = None, user_id: Optional[str] = None,
                   status: Optional[str] = None) -> list[tuple[Page, str]]:
        out = []
        for page, body in self.iter_pages():
            if type and page.type != type:
                continue
            if user_id and page.user_id != user_id:
                continue
            if status and page.status.value != status:
                continue
            out.append((page, body))
        return out

    # ---------------- raw sources (immutable) ----------------

    def append_raw(self, text: str, source_type: str = "conversation", origin: str = "",
                   user_id: Optional[str] = None, agent_id: Optional[str] = None,
                   run_id: Optional[str] = None, metadata: Optional[dict] = None,
                   pii: Optional[list[str]] = None) -> RawEntry:
        created = utcnow_iso()
        scope = self.user_prefix(user_id) if user_id else "g"
        stamp = _TS_SAFE.sub("-", created)
        name = f"{stamp}__{slugify(origin, max_len=40, fallback='entry')}"
        directory = self.raw_dir / scope / slugify(source_type, fallback="conversation")
        directory.mkdir(parents=True, exist_ok=True)
        path, n = directory / f"{name}.md", 2
        while path.exists():                          # same-second collisions
            path, n = directory / f"{name}-{n}.md", n + 1
        entry = RawEntry(
            path=self.rel(path), text=text, source_type=source_type, origin=origin,
            user_id=user_id, agent_id=agent_id, run_id=run_id,
            metadata=metadata or {}, created=created, pii=pii or [],
        )
        meta = entry.model_dump(exclude={"path", "text"}, exclude_none=True, exclude_defaults=True)
        meta["created"] = created
        path.write_text(f"---\n{_yaml_dump(meta)}---\n\n{text}\n", encoding="utf-8")
        return entry

    def read_raw(self, rel_path: str) -> Optional[RawEntry]:
        path = (self.root / rel_path).resolve()
        if self.root not in path.parents or not path.exists():
            return None
        post = frontmatter.loads(path.read_text(encoding="utf-8"))
        meta = dict(post.metadata)
        meta.pop("path", None)
        return RawEntry(path=rel_path, text=post.content, **{
            k: v for k, v in meta.items() if k in RawEntry.model_fields
        })

    def write_failed(self, payload: str, reason: str) -> Path:
        directory = self.raw_dir / "failed"
        directory.mkdir(parents=True, exist_ok=True)
        stamp = _TS_SAFE.sub("-", utcnow_iso())
        path = directory / f"{stamp}__{slugify(reason, max_len=40, fallback='error')}.md"
        n = 2
        while path.exists():
            path = directory / f"{stamp}__{slugify(reason, max_len=40)}-{n}.md"
            n += 1
        path.write_text(payload, encoding="utf-8")
        return path

    # ---------------- index.md ----------------

    def regenerate_index(self) -> None:
        """Rebuild index.md by parsing every page — the from-truth path (init,
        import, erasure, recovery). The write pipeline regenerates it from the
        SQLite index instead (same format, O(changed) not O(N))."""
        entries = [(p.type, p.id, p.title, p.summary, p.confidence, p.status.value)
                   for p, _ in self.iter_pages()]
        self.index_path.write_text(format_index(entries), encoding="utf-8")

    def index_summary(self, max_chars: int = 4000) -> str:
        """Compact page inventory fed to extractors so they route to existing
        pages instead of sprawling duplicates (§10.7). O(N) markdown parse —
        the pipeline uses the SQLite-backed equivalent when wired by Memory."""
        lines = []
        for page, _ in self.iter_pages():
            if page.status.value != "active":
                continue
            tags = ",".join(page.tags[:5])
            subjects = ",".join(sorted(list({c.subject for c in page.active_claims() if c.subject})))
            lines.append(f"{page.id} | {page.title} | {tags} | {subjects}")
        text = "\n".join(lines)
        return text[:max_chars]


def format_index(entries: list[tuple]) -> str:
    """Render index.md from (type, id, title, summary, confidence, status)
    tuples. One formatter for both the markdown-truth and SQLite paths so the
    two never drift byte-wise."""
    by_type: dict[str, list[tuple]] = {}
    for e in entries:
        by_type.setdefault(e[0], []).append(e)
    lines = [
        "# Memory Index", "",
        "Auto-generated by MarkMem after every write — do not edit by hand.", "",
    ]
    for ptype in sorted(by_type):
        lines.append(f"## {ptype}")
        for _, pid, title, summary, confidence, status in sorted(by_type[ptype], key=lambda e: e[1]):
            flags = "" if status == "active" else f", {status}"
            summary = (summary or "").replace("\n", " ").strip()
            lines.append(f"- `{pid}` — {title} — {summary} (conf {confidence:.2f}{flags})")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
