"""MarkMem demo chatbot — Streamlit UI over the Memory layer + Nemotron.

Run from the project root:

    streamlit run chatbot/app.py

Each turn: packed memory context is retrieved for the user's message, injected
into Nemotron's system prompt, the reply is generated, and the whole turn is
written back into memory (compiled synchronously by default so the next turn
already sees it). The sidebar exposes the memory internals: stats, the user's
live claim ledger, the review queue, and maintenance actions.

Without an NVIDIA_API_KEY the app runs in a labelled no-LLM demo mode — memory
ingestion/retrieval is fully real; only reply generation is stubbed.
"""
from __future__ import annotations

import streamlit as st

try:                                    # `streamlit run chatbot/app.py` (script dir on path)
    from llm import NemotronClient
except ImportError:                     # `python -m streamlit run ...` from elsewhere
    from chatbot.llm import NemotronClient

from markmem import Memory

PERSONA = (
    "You are a helpful personal assistant with long-term memory. "
    "A '### Memory' section may follow with facts about this user — each block "
    "carries an id, a confidence score, and a provenance tag. Rely on active facts, "
    "prefer user_stated over inferred ones, never treat facts marked "
    "[superseded ...] as current, and mention the memory id when a remembered "
    "fact drives your answer. If memory is empty, just say you don't know yet. "
    "/no_think"   # disables Nemotron reasoning mode; inert text for other models
)


@st.cache_resource(show_spinner="Opening memory repo…")
def get_memory(repo_path: str) -> Memory:
    return Memory(repo_path=repo_path, start_worker=True)


@st.cache_resource
def get_llm() -> NemotronClient:
    return NemotronClient()


def demo_reply(context: str) -> str:
    """No-LLM fallback: proves the memory loop without generating prose."""
    facts = [ln for ln in context.splitlines() if ln.startswith("- ")]
    if facts:
        return ("**(demo mode — no NVIDIA_API_KEY set)** Here is what memory "
                "injected for this turn:\n" + "\n".join(facts[:8]))
    return ("**(demo mode — no NVIDIA_API_KEY set)** No compiled memory for this "
            "user yet — tell me something about yourself.")


def render_sidebar(memory: Memory, user_id: str) -> dict:
    st.sidebar.title("🧠 MarkMem")
    st.sidebar.caption(f"repo: `{memory.repo.root}`")

    settings = {
        "sync": st.sidebar.checkbox(
            "Compile each turn (flush)", value=True,
            help="Off = compilation stays async in the background worker; "
                 "the next turn may not see this turn's facts yet."),
        "top_k": st.sidebar.slider("Search top-k", 1, 10, 5),
        "show_context": st.sidebar.checkbox("Show injected memory per reply", value=True),
    }

    col1, col2 = st.sidebar.columns(2)
    if col1.button("Flush queue"):
        n = memory.flush()
        st.sidebar.success(f"compiled {n} entries")
    if col2.button("Maintenance"):
        report = memory.maintenance()
        st.sidebar.json({k: len(v) for k, v in report.items()})

    with st.sidebar.expander("📊 Memory stats"):
        st.json(memory.stats())

    with st.sidebar.expander(f"🪪 Claim ledger — {user_id}"):
        profile = memory.get(f"{memory.repo.user_prefix(user_id)}/user/profile")
        if profile and profile["claims"]:
            for c in profile["claims"]:
                mark = "✅" if not c.get("valid_until") else f"⏸ until {c['valid_until']}"
                st.markdown(f"{mark} **{c.get('subject') or '—'}** · {c['text']}  \n"
                            f"<small>{c['provenance']} · conf {c['confidence']:.2f}</small>",
                            unsafe_allow_html=True)
        else:
            st.caption("no compiled claims yet")

    review_items = memory.pipeline.review_queue.list()
    with st.sidebar.expander(f"🔍 Review queue ({len(review_items)})"):
        for item in review_items:
            st.markdown(f"**{item['id']}** — {', '.join(item['reasons'])}")
            st.code(item["op"].get("summary") or item["op"].get("body", "")[:200])
            a, r = st.columns(2)
            if a.button("accept", key=f"acc-{item['id']}"):
                memory.pipeline.review_accept(item["id"])
                st.rerun()
            if r.button("reject", key=f"rej-{item['id']}"):
                memory.pipeline.review_reject(item["id"])
                st.rerun()
        if not review_items:
            st.caption("empty — nothing quarantined")
    return settings


def main() -> None:
    st.set_page_config(page_title="MarkMem Chat", page_icon="🧠", layout="wide")

    with st.sidebar:
        repo_path = st.text_input("Memory repo path", value="./chat-memory")
        user_id = st.text_input("User id", value="demo")

    memory = get_memory(repo_path)
    llm = get_llm()
    settings = render_sidebar(memory, user_id)

    st.title("MarkMem memory chatbot")
    if llm.available:
        st.caption(f"model: `{llm.model}` · memory: git-native markdown at `{repo_path}`")
    else:
        st.warning("No NVIDIA_API_KEY in .env — running in no-LLM demo mode. "
                   "Memory ingestion/retrieval is fully live; only replies are stubbed.")

    chat_key = f"chat::{repo_path}::{user_id}"
    history = st.session_state.setdefault(chat_key, [])

    for turn in history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("context") and settings["show_context"]:
                with st.expander("memory injected for this turn"):
                    st.code(turn["context"] or "(none)", language="markdown")

    prompt = st.chat_input("Say something…")
    if not prompt:
        return

    history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    context = memory.search(prompt, user_id=user_id, top_k=settings["top_k"],
                            format="context")
    with st.chat_message("assistant"):
        with st.spinner("thinking…"):
            if llm.available:
                recent = [{"role": t["role"], "content": t["content"]}
                          for t in history[-10:]]
                system = PERSONA + ("\n\n" + context if context else "")
                try:
                    reply = llm.chat(system, recent)
                except Exception as e:
                    reply = f"⚠️ LLM call failed: {e}"
            else:
                reply = demo_reply(context)
        st.markdown(reply)
        if settings["show_context"]:
            with st.expander("memory injected for this turn"):
                st.code(context or "(none)", language="markdown")

    history.append({"role": "assistant", "content": reply, "context": context})

    memory.add(f"user: {prompt}\nassistant: {reply}", user_id=user_id,
               origin="streamlit")
    if settings["sync"]:
        memory.flush()


if __name__ == "__main__":
    main()
