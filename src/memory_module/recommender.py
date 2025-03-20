import os
import random
from openai import OpenAI
from dotenv import load_dotenv
from groq import Groq
from memory_module.db import update_customer_data, get_customer_profile

load_dotenv()
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def recommend(last_input, customer_data):
    # prompt = f"""
    # You're in the middle of a conversation with the customer. This is what you know about them.
    # {customer_data}
    
    
    # the last thing they told you is: {last_input}
    
    # Try to be enthusiastic and funny in general, but you should consider their mood. You are context-aware of them, through the provided information. If the customer hasn't ordered food yet, make suggestions based on their preferences.
    # Try to make them feel special, and remember the customer is always right. However don't talk too much, make your answers smooth, concise and precise

    # """
    
    prompt = f"""
    You're in the middle of a conversation with the customer. This is what you know about them.
    {customer_data}
    
    
    the last thing they told you is: {last_input}
    
    I need you to return a JSON object in exactly this format 
    {{'memories': '<MEMORY>'}} 
    
    where MEMORY is the a simple short string describing the customer data combined with the last input
    """

    response = client.chat.completions.create(model="llama-3.3-70b-versatile",  # or "gpt-4" if available
    messages=[
        {"role": "system", "content": "You summarize restaurant customer interactions concisely."},
        {"role": "user", "content": prompt}
    ],
    max_tokens=300,
    temperature=0.2)

    summary = response.choices[0].message.content.strip()
    return summary

