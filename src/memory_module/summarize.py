from openai import OpenAI
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def summarize_conversation(conversation_text):
    prompt = f"""
    Summarize the following restaurant interaction concisely, highlighting customer preferences, dietary restrictions, likes or dislikes, and any important details.

    Conversation:
    {conversation_text}

    Concise summary:
    """

    response = client.chat.completions.create(model="llama-3.3-70b-versatile",  # or "gpt-4" if available
    messages=[
        {"role": "system", "content": "You summarize restaurant customer interactions concisely."},
        {"role": "user", "content": prompt}
    ],
    max_tokens=100,
    temperature=0.2)

    summary = response.choices[0].message.content.strip()
    return summary


user_input = f"""Customer: Last week I had your spicy ramen, it was fantastic. Today I'd prefer something lighter—maybe seafood? But please, no shrimp, I'm allergic.
Robot: Noted—seafood, no shrimp. Would you like our grilled salmon?
Customer: Salmon sounds perfect."""

print(summarize_conversation(user_input))