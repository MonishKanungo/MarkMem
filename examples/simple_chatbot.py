"""The entire integration pattern: search -> reply -> add, once per turn.

Run:  python examples/simple_chatbot.py
Works fully offline (heuristic extractor). Point `call_llm` at a real model
and set ANTHROPIC_API_KEY for LLM-compiled memory.
"""
from strata import Memory

memory = Memory(repo_path="./chat-memory")


def call_llm(user_message: str, context: str) -> str:
    """Stand-in for your actual LLM call — the packed context goes into the
    system prompt; blocks carry ids/confidence/provenance so the model can cite."""
    lines = [ln for ln in context.splitlines() if ln.startswith("- ")]
    if lines:
        return f"(demo reply) Here's what I remember about you:\n" + "\n".join(lines[:5])
    return "(demo reply) I don't know you yet — tell me about yourself!"


def chat_turn(user_message: str, user_id: str) -> str:
    context = memory.search(user_message, user_id=user_id, format="context")
    reply = call_llm(user_message, context)
    memory.add(f"user: {user_message}\nassistant: {reply}", user_id=user_id)
    # Compilation is asynchronous: normally the background worker picks this up
    # within seconds and add() costs ~ms. This demo flushes each turn so the
    # very next turn already sees compiled memory.
    memory.flush()
    return reply


if __name__ == "__main__":
    for turn in ("Hi! I'm vegetarian and I prefer window seats.",
                 "Actually, I prefer aisle seats now.",
                 "What do you know about my seating preference?"):
        print(f"\n> {turn}")
        print(chat_turn(turn, user_id="demo-user"))
    print("\n--- packed memory after the conversation ---")
    print(memory.search("preferences", user_id="demo-user", format="context"))
    memory.close()
