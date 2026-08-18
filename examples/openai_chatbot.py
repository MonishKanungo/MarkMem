"""MarkMem + OpenAI — drop-in memory for any OpenAI-compatible chatbot.

Install:
    pip install markmem openai

Run:
    OPENAI_API_KEY=sk-... python examples/openai_chatbot.py
"""
import os
from openai import OpenAI
from markmem import Memory

client = OpenAI()
memory = Memory(repo_path="./chat-memory")

SYSTEM = (
    "You are a helpful assistant with long-term memory. "
    "A '### Memory' block may follow with facts about this user. "
    "Cite memory ids when you rely on them. "
    "If memory is empty, say you don't know yet."
)


def chat(user_message: str, user_id: str) -> str:
    # 1. Retrieve relevant memory (packed, token-budgeted)
    context = memory.search(user_message, user_id=user_id, format="context")
    system = SYSTEM + ("\n\n" + context if context else "")

    # 2. Call OpenAI with memory injected into system prompt
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user_message},
        ],
    )
    reply = response.choices[0].message.content

    # 3. Store the turn — auto-captured, PII-gated, async compiled
    memory.add(f"user: {user_message}\nassistant: {reply}", user_id=user_id)
    memory.flush()  # remove this line for async compilation
    return reply


if __name__ == "__main__":
    uid = "alice"
    turns = [
        "Hi! I'm Alice. I'm a vegetarian and prefer window seats on flights.",
        "Actually I switched to aisle seats last month.",
        "What do you know about my travel preferences?",
    ]
    for msg in turns:
        print(f"\nUser: {msg}")
        print(f"Bot:  {chat(msg, uid)}")

    memory.close()
