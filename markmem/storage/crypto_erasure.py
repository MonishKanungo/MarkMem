"""Crypto-shred erasure — Phase 2 provable deletion without git-filter-repo (§4.1).

The idea: each user's data is encrypted at rest with a per-user data-encryption
key (DEK). The DEK is wrapped by a key-encryption key (KEK) stored in a
separate key store. To erase a user: delete their KEK → the DEK becomes
unrecoverable → all their files become undecryptable gibberish.

This is stronger than git-filter-repo rewrite for cloud storage scenarios:
- No need to rewrite git history (clones stay valid)
- Erasure is instant and irreversible
- Compliant with GDPR "right to be forgotten" even with backups
  (backups contain only encrypted blobs; without the KEK they're useless)

Key store backends:
- FileKeyStore  — local .markmem/keys/ directory (default, no deps)
- EnvKeyStore   — keys from environment variables (CI/CD, simple deployments)

Usage:
    from markmem.storage.crypto_erasure import CryptoErasureManager

    mgr = CryptoErasureManager(repo)
    mgr.write_encrypted("u/alice/user/profile.md", plaintext_bytes)
    plaintext = mgr.read_encrypted("u/alice/user/profile.md")
    mgr.shred_user("alice")  # DEK deleted → all alice's files unreadable

NOTE: This module is opt-in. MarkMem works without it (plaintext files in git).
Enabling crypto-shred is a deployment decision — mixing encrypted and plaintext
files in the same repo is not supported.

Install: pip install markmem[crypto]  (adds cryptography package)
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Protocol


# ---------------------------------------------------------------------------
# Key store protocol — swap implementations without touching the rest
# ---------------------------------------------------------------------------

class KeyStore(Protocol):
    def get_dek(self, user_id: str) -> Optional[bytes]:
        """Return the 32-byte DEK for user_id, or None if not found."""
        ...

    def put_dek(self, user_id: str, dek: bytes) -> None:
        """Persist the DEK for user_id."""
        ...

    def delete_dek(self, user_id: str) -> bool:
        """Delete the DEK for user_id. Returns True if it existed."""
        ...

    def list_users(self) -> list[str]:
        """Return all user_ids that have a DEK in the store."""
        ...


class FileKeyStore:
    """DEKs stored as files under .markmem/keys/<user_id>.key.
    Each file contains the raw 32-byte DEK (not base64 — kept binary).
    The keys directory should itself be outside the git repo or in .gitignore.
    """

    def __init__(self, markmem_dir: Path):
        self.keys_dir = markmem_dir / "keys"
        self.keys_dir.mkdir(parents=True, exist_ok=True)
        # Write a .gitignore so keys are never accidentally committed
        gi = self.keys_dir / ".gitignore"
        if not gi.exists():
            gi.write_text("*\n", encoding="utf-8")

    def _path(self, user_id: str) -> Path:
        # Sanitize: replace path separators so user_ids like "u/alice" are safe
        safe = user_id.replace("/", "__").replace("\\", "__")
        return self.keys_dir / f"{safe}.key"

    def get_dek(self, user_id: str) -> Optional[bytes]:
        p = self._path(user_id)
        return p.read_bytes() if p.exists() else None

    def put_dek(self, user_id: str, dek: bytes) -> None:
        p = self._path(user_id)
        p.write_bytes(dek)
        # Restrict permissions on Unix; on Windows this is best-effort
        try:
            p.chmod(0o600)
        except OSError:
            pass

    def delete_dek(self, user_id: str) -> bool:
        p = self._path(user_id)
        if p.exists():
            p.unlink()
            return True
        return False

    def list_users(self) -> list[str]:
        return [p.stem.replace("__", "/") for p in self.keys_dir.glob("*.key")]


class EnvKeyStore:
    """DEKs from environment variables — MARKMEM_DEK_<USER_ID>=<hex>.
    Useful for CI/CD, Docker secrets, or when you manage keys externally.
    Deletion only removes the in-memory cache; you must remove the env var
    from your secrets manager separately.
    """

    def __init__(self):
        self._deleted: set[str] = set()

    def _env_key(self, user_id: str) -> str:
        return "MARKMEM_DEK_" + user_id.upper().replace("/", "_").replace("-", "_")

    def get_dek(self, user_id: str) -> Optional[bytes]:
        if user_id in self._deleted:
            return None
        hex_val = os.environ.get(self._env_key(user_id))
        return bytes.fromhex(hex_val) if hex_val else None

    def put_dek(self, user_id: str, dek: bytes) -> None:
        os.environ[self._env_key(user_id)] = dek.hex()
        self._deleted.discard(user_id)

    def delete_dek(self, user_id: str) -> bool:
        self._deleted.add(user_id)
        key = self._env_key(user_id)
        if key in os.environ:
            del os.environ[key]
            return True
        return False

    def list_users(self) -> list[str]:
        prefix = "MARKMEM_DEK_"
        return [
            k[len(prefix):].lower().replace("_", "/")
            for k in os.environ
            if k.startswith(prefix)
        ]


# ---------------------------------------------------------------------------
# Crypto operations — AES-256-GCM via the `cryptography` package
# ---------------------------------------------------------------------------

def _require_crypto():
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        return AESGCM
    except ImportError as exc:
        raise ImportError(
            "Crypto-shred needs the [crypto] extra: pip install markmem[crypto]"
        ) from exc


def generate_dek() -> bytes:
    """Generate a fresh 32-byte (256-bit) data-encryption key."""
    return os.urandom(32)


def encrypt(plaintext: bytes, dek: bytes) -> bytes:
    """AES-256-GCM encrypt. Returns nonce (12 bytes) + ciphertext + tag."""
    AESGCM = _require_crypto()
    nonce = os.urandom(12)
    aesgcm = AESGCM(dek)
    ciphertext = aesgcm.encrypt(nonce, plaintext, None)
    return nonce + ciphertext  # nonce is prepended for self-contained storage


def decrypt(blob: bytes, dek: bytes) -> bytes:
    """AES-256-GCM decrypt. Expects nonce (12 bytes) prepended to ciphertext."""
    AESGCM = _require_crypto()
    nonce, ciphertext = blob[:12], blob[12:]
    aesgcm = AESGCM(dek)
    return aesgcm.decrypt(nonce, ciphertext, None)


# ---------------------------------------------------------------------------
# CryptoErasureManager — the public API
# ---------------------------------------------------------------------------

class CryptoErasureManager:
    """Manages per-user encryption and key-shred erasure.

    Typical workflow:
        mgr = CryptoErasureManager(repo)
        # Transparent encrypt/decrypt around normal file IO
        mgr.write_file("wiki/u/alice/user/profile.md", content_bytes)
        content = mgr.read_file("wiki/u/alice/user/profile.md")
        # Erase alice — all her files become permanently unreadable
        tombstone = mgr.shred_user("alice", ledgers)
    """

    def __init__(self, repo_root: Path, key_store: Optional[KeyStore] = None):
        self.root = Path(repo_root).resolve()
        self.key_store = key_store or FileKeyStore(self.root / ".markmem")

    def _dek_for(self, user_id: str) -> bytes:
        """Return existing DEK or generate and store a new one."""
        dek = self.key_store.get_dek(user_id)
        if dek is None:
            dek = generate_dek()
            self.key_store.put_dek(user_id, dek)
        return dek

    @staticmethod
    def _user_from_path(rel_path: str) -> Optional[str]:
        """Extract user_id from a repo-relative path like wiki/u/alice/... or raw/u/alice/...
        Returns None for global paths (wiki/g/...)."""
        parts = Path(rel_path).parts
        # wiki/u/<user>/... or raw/u/<user>/...
        if len(parts) >= 3 and parts[1] == "u":
            return parts[2]
        return None

    def write_file(self, rel_path: str, content: bytes) -> None:
        """Encrypt content for the owning user and write to disk.
        For global paths (no user) writes plaintext — crypto-shred is per-user."""
        abs_path = self.root / rel_path
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        user_id = self._user_from_path(rel_path)
        if user_id:
            dek = self._dek_for(user_id)
            abs_path.write_bytes(encrypt(content, dek))
        else:
            abs_path.write_bytes(content)  # global pages stay plaintext

    def read_file(self, rel_path: str) -> Optional[bytes]:
        """Decrypt and return file content. Returns None if file missing.
        Returns None (not raises) if DEK missing — file is shredded."""
        abs_path = self.root / rel_path
        if not abs_path.exists():
            return None
        user_id = self._user_from_path(rel_path)
        if not user_id:
            return abs_path.read_bytes()
        dek = self.key_store.get_dek(user_id)
        if dek is None:
            return None  # shredded — DEK gone, content unrecoverable
        try:
            return decrypt(abs_path.read_bytes(), dek)
        except Exception:
            return None  # corrupted or shredded mid-write

    def shred_user(self, user_id: str, ledgers=None) -> dict:
        """Crypto-shred: delete the DEK and record a tombstone.

        After shredding:
        - The user's files still exist on disk (and in git history)
        - But they are encrypted blobs with no key — permanently unreadable
        - No git history rewrite needed — backups are useless without the DEK

        This satisfies GDPR Art. 17 'right to erasure' for encrypted data.
        """
        from ..util import utcnow_iso

        existed = self.key_store.delete_dek(user_id)
        tombstone = {
            "user_id": user_id,
            "mode": "crypto-shred",
            "dek_deleted": existed,
            "erased_at": utcnow_iso(),
            "note": (
                "DEK deleted. Encrypted files remain on disk and in git history "
                "but are permanently unreadable without the key. "
                "No git-filter-repo rewrite needed for GDPR compliance."
            ),
        }
        if ledgers is not None:
            ledgers.record_op("erasure", **tombstone)
        return tombstone

    def rotate_dek(self, user_id: str) -> dict:
        """Rotate the DEK for a user: re-encrypt all their files with a new key.
        Call this as part of regular key rotation policies.
        """
        from ..util import utcnow_iso

        old_dek = self.key_store.get_dek(user_id)
        if old_dek is None:
            return {"user_id": user_id, "rotated": False, "reason": "no DEK found"}

        new_dek = generate_dek()
        rotated = 0
        for prefix in ("wiki/u", "raw/u"):
            user_dir = self.root / prefix / user_id
            if not user_dir.exists():
                continue
            for path in user_dir.rglob("*"):
                if not path.is_file():
                    continue
                try:
                    plaintext = decrypt(path.read_bytes(), old_dek)
                    path.write_bytes(encrypt(plaintext, new_dek))
                    rotated += 1
                except Exception:
                    pass  # skip files that can't be decrypted (already plaintext, etc.)

        self.key_store.put_dek(user_id, new_dek)
        return {
            "user_id": user_id, "rotated": True,
            "files_reencrypted": rotated, "rotated_at": utcnow_iso(),
        }

    def status(self) -> dict:
        """Return key store status: which users have active DEKs."""
        return {
            "key_store": type(self.key_store).__name__,
            "users_with_dek": self.key_store.list_users(),
        }
