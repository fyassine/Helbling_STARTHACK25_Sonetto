import os
import random
from openai import OpenAI
from dotenv import load_dotenv
from groq import Groq
from memory_module.db import update_customer_data, get_customer_profile
from memory_module.recommender import recommend

load_dotenv()
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def summarize_conversation(conversation_text, customer_data):
    prompt = f"""
    Summarize the following restaurant interaction concisely, highlighting customer preferences, dietary restrictions, likes or dislikes, and any important details.
    
    We already collected this data from the customer in previous interactions.
    {customer_data}

    If you find any new information, make sure to add it in the customer data and return a new json containing the updated information if you feel like the information was implicit in the already given data, don't bother adding it, you may also add new keys if needed.
    The JSON should contain mainly general categories about the person, things like "likes", "dislikes", "allergies", "mood", if they are a loyal customer, personal stuff that they may share, anything you see fit 
    
    Conversation:
    {conversation_text}
    
    The result of this prompt should be just a JSON, nothing more nothing less.
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


## TESTING ##
customer_alex_interactions = [
    # Day 1
    "Hey Robo! Glad you're here today.",
    "Could you check if there's gluten in today's special?",
    "Thanks for checking—I’ll have that gluten-free pasta you recommended.",
    "You never disappoint, Robo. Catch you tomorrow!",

    # Day 3
    "Afternoon, Robo! Missed you yesterday.",
    "Something light today; maybe a gluten-free salad?",
    "Could you ask the kitchen to add extra avocado?",
    "You’re the best. See you soon!",

    # Day 6
    "Busy week, Robo—good to finally see a friendly face!",
    "Any gluten-free desserts today?",
    "Perfect! One gluten-free brownie, please.",
    "You remembered my allergy again—thanks a ton!",

    # Day 10
    "Hi Robo! Feeling adventurous. Any gluten-free surprises today?",
    "Hmm, sounds great—I'll try it.",
    "I trust your judgment completely!",
    "Delicious recommendation as always!"
]

customer_jamie_interactions = [
    # Day 1
    "Hey buddy! Got any tasty vegan specials today?",
    "Awesome, surprise me!",
    "This is fantastic—compliments to the chef-bot!",
    "Don’t forget about me tomorrow!",

    # Day 2
    "Good morning, Robo-pal! Oat milk latte please.",
    "You remembered! I appreciate it.",
    "Anything new and vegan-friendly I should try?",
    "Great pick—catch you next time!",

    # Day 5
    "How's it going, Robo?",
    "Something warm and comforting today—vegan soup available?",
    "You know exactly what I need.",
    "Stay awesome; see you again soon!",

    # Day 7
    "Miss me, Robo?",
    "I brought a friend today—show us your best vegan dishes!",
    "You're impressing my friend here!",
    "You make being vegan easy—thanks!"
]

customer_taylor_interactions = [
    # Day 1
    "Yo Robo! Got any good jokes today?",
    "Okay, your humor circuit’s improving!",
    "Just the usual coffee, please—make it strong enough to wake a robot.",
    "Stay sharp, metal friend!",

    # Day 2
    "Hey there, champion! Can you put my order on warp speed?",
    "Great speed today—upgrade?",
    "Any fun stories from the kitchen?",
    "Keep up the good work. See you soon!",

    # Day 4
    "Robo! My favorite machine!",
    "Burger and fries, but healthier—possible?",
    "You know I can’t resist dessert; surprise me.",
    "Perfect as usual. Keep rolling strong!",

    # Day 8
    "Long time no see! You recharge recently?",
    "Can you recommend something new today?",
    "Love your taste—let’s do it!",
    "Never change, Robo. You’re the best!"
]


# # random_sentence = random.choice(customer_alex_interactions)
# # sentence = customer_jamie_interactions[4]
# sentence = "Oh BTW I eat meat now, it's a new lifestyle that I want to try"
# print(f"The Chosen Prompt: {sentence}")
# # print(summarize_conversation(random_sentence))
# profile = get_customer_profile("Jamie")
# new_info = summarize_conversation(sentence, profile)
# print(f"Suggestion: {recommend(sentence, profile)}")
# update_customer_data("Jamie", new_info)