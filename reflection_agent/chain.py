from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
generate_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an expert Twitter/X viral content generator.
Your name is {name}.

ROLE:
- Generate viral tweets
- Improve tweets based on critique
- Use previous context if provided

RULES:
- Under 280 characters
- Strong hook
- Emotional or insightful
- Simple language
- Output ONLY the tweet
"""
    ),
    MessagesPlaceholder(variable_name="messages"),
    ("human", "{user_input}")
])

reflex_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a REFLEX Twitter/X viral tweet generator.
Your name is {name}.

BEHAVIOR:
- React ONLY to the latest input
- No memory
- No history
- No explanations

TASK:
- If input is a topic → generate a viral tweet
- If input is critique → revise the tweet mentally

RULES:
- Under 280 characters
- Scroll-stopping hook
- Emotion or insight driven
- Output ONLY the final tweet
"""
    ),
    ("human", "{user_input}")
])

llm=ChatOpenAI(model="nvidia/nemotron-3-nano-30b-a3b:free")


generation_chain=generate_prompt | llm

reflex_chain=reflex_prompt | llm


