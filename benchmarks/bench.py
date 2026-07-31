"""Honest latency benchmark (§9.10): p50/p95 for add() and search() at N pages.

Run:  python benchmarks/bench.py [n_pages]
Reproducible, offline, commodity-CPU numbers — no vendor magic.
"""
import statistics
import sys
import tempfile
import time
from pathlib import Path

from strata import Memory

TOPICS = ["terraform", "kubernetes", "postgres", "lambda", "budget", "onboarding",
          "migration", "audit", "billing", "search"]


def pct(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * p))]


def main(n_pages: int = 1000) -> None:
    root = Path(tempfile.mkdtemp()) / "bench-mem"
    m = Memory(repo_path=root, start_worker=False)
    print(f"repo: {root}  target pages: ~{n_pages}")

    add_times = []
    t_all = time.perf_counter()
    for i in range(n_pages):
        topic = TOPICS[i % len(TOPICS)]
        t0 = time.perf_counter()
        m.add(f"Note {i}: the {topic} rollout needs review of item {i}.",
              user_id=f"user-{i % 20}")
        add_times.append((time.perf_counter() - t0) * 1000)
        if (i + 1) % 200 == 0:
            m.flush()           # batch-compile as we go, like the worker would
    m.flush()
    ingest_s = time.perf_counter() - t_all

    search_times = []
    for i in range(200):
        topic = TOPICS[i % len(TOPICS)]
        t0 = time.perf_counter()
        m.search(f"{topic} rollout review", user_id=f"user-{i % 20}")
        search_times.append((time.perf_counter() - t0) * 1000)

    pack_times = []
    for i in range(50):
        t0 = time.perf_counter()
        m.search("rollout review status", user_id=f"user-{i % 20}", format="context")
        pack_times.append((time.perf_counter() - t0) * 1000)

    stats = m.stats()
    print(f"pages: {sum(stats['pages_by_type'].values())}  claims: {stats['claims']}")
    print(f"ingest wall time (incl. compile+commit): {ingest_s:.1f}s "
          f"({n_pages / ingest_s:.0f} adds/s end-to-end)")
    print(f"add() enqueue      p50 {pct(add_times, .5):6.1f} ms   p95 {pct(add_times, .95):6.1f} ms")
    print(f"search() L1        p50 {pct(search_times, .5):6.1f} ms   p95 {pct(search_times, .95):6.1f} ms")
    print(f"search(context)    p50 {pct(pack_times, .5):6.1f} ms   p95 {pct(pack_times, .95):6.1f} ms")
    m.close()


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1000)
