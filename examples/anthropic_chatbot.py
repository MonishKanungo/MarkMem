"""MarkMem + Anthropic Claude — memory-augmented Claude chatbot.

Install:
    pip install markmem anthropic
    pip install markmem[llm]   # also activates LLM extraction

Run:
    ANTHROPIC_API_KEY=sk-ant-... python examples/anthropic_chatbot.py
"""
import os
import anthropic
from markmem import Memory

client = anthropic.Anthropic()
memory = Memory(repo_path="./chat-memory")  # LLM extractor auto-activates with API key

SYSTEM = (
    "You are a helpful assistant with long-term memory. "
    "A '### Memory' block may follow with facts about this user. "
    "Cite the memory id in brackets when a fact drives your answer."
)


def chat(user_message: str, user_id: str) -> str:
    context = memory.search(user_message, user_id=user_id, format="context")
    system = SYSTEM + ("\n\n" + context if context else "")

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user_message}],
    )
    reply = response.content[0].text

    memory.add(f"user: {user_message}\nassistant: {reply}", user_id=user_id)
    memory.flush()
    return reply


if __name__ == "__main__":
    uid = "bob"
    for msg in [
        "I'm Bob. I work at NVIDIA as a software engineer and I love Python.",
        "I just switched jobs — now I'm at Google working on Gemini.",
        "Where do I work?",
    ]:
        print(f"\nUser: {msg}")
        print(f"Bot:  {chat(msg, uid)}")

    memory.close()
