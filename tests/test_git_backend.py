import threading

import pytest

from strata.storage.git_backend import SubprocessGit, writer_lock
from strata.storage.repo import Repo


@pytest.fixture()
def git_repo(tmp_path):
    repo = Repo.scaffold(tmp_path / "r")
    git = SubprocessGit(repo.root)
    git.init()
    return repo, git


def test_init_idempotent_and_identity(git_repo):
    repo, git = git_repo
    git.init()                                    # second init must not fail
    assert (repo.root / ".git").exists()
    name = git._run("config", "user.name").stdout.strip()
    assert name                                   # some identity always configured


def test_commit_all_batches_and_skips_clean(git_repo):
    repo, git = git_repo
    sha1 = git.commit_all("first")
    assert sha1
    assert git.commit_all("nothing to do") is None
    (repo.root / "wiki" / "x.md").write_text("x", encoding="utf-8")
    (repo.root / "wiki" / "y.md").write_text("y", encoding="utf-8")
    sha2 = git.commit_all("two files, one commit")
    assert sha2 and sha2 != sha1
    assert len(git.history()) == 2


def test_history_follow_and_diff(git_repo):
    repo, git = git_repo
    page = repo.wiki_dir / "page.md"
    page.write_text("v1", encoding="utf-8")
    git.commit_all("add page")
    page.write_text("v2", encoding="utf-8")
    git.commit_all("edit page")
    commits = git.history("wiki/page.md")
    assert [c.message for c in commits] == ["edit page", "add page"]
    with_diff = git.history("wiki/page.md", include_diff=True)
    assert "v2" in with_diff[0].diff


def test_history_empty_repo(tmp_path):
    repo = Repo.scaffold(tmp_path / "fresh")
    git = SubprocessGit(repo.root)
    git.init()
    assert git.history() == []


def test_rm_ignores_missing(git_repo):
    repo, git = git_repo
    git.commit_all("base")
    git.rm(["wiki/nonexistent"])                  # must not raise


def test_writer_lock_serializes(git_repo):
    repo, _ = git_repo
    order = []

    def hold(name):
        with writer_lock(repo.root, timeout=10):
            order.append(f"{name}-in")
            import time
            time.sleep(0.15)
            order.append(f"{name}-out")

    t1 = threading.Thread(target=hold, args=("a",))
    t2 = threading.Thread(target=hold, args=("b",))
    t1.start(); t2.start(); t1.join(); t2.join()
    # no interleaving: each -in is immediately followed by its own -out
    assert order[0].split("-")[0] == order[1].split("-")[0]
    assert order[2].split("-")[0] == order[3].split("-")[0]
