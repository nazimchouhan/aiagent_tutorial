from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import datetime
from langchain_core.messages import AIMessage,HumanMessage
from schema import AnswerQuestion,ReviseAnswer
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers.openai_tools import PydanticToolsParser

from dotenv import load_dotenv

load_dotenv()
# Actor agent prompt


pydantic_parser=PydanticToolsParser(tools=[AnswerQuestion])
pydantic_answer=PydanticToolsParser(tools=[ReviseAnswer])

actor_prompt_template=ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert AI Researcher.
Current time:{time}
1. {first_instructions}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. After the reflection, **list 1-3 search queries separately** for
researching improvements. Do not include them inside the reflection            
"""
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", " Answer the user's question above using the required format.")

    ]

).partial(
    time=lambda:datetime.datetime.now().isoformat(),
)

# Revisor Instructions
revise_instructions="""Revise your previous answer using the new information.
-You should use the previous critique to add important information to your answer 
-You MUST include numerical citations in your revised answer to ensure it can be verified .
-Add a "References" section to the bottom of your answer (which does not count towords the word limit).
    In form of:
            - [1] https://example.com
            - [2] https://example.com
-You should use the previous critique to remove superfluous information from your answer and make SURE and make sure it is not more than 250 words.
"""
first_responder_prompt=actor_prompt_template.partial(
    first_instructions="provide a detailed ~250 word answer"
)

first_revisor_prompt=actor_prompt_template.partial(
    first_instructions=revise_instructions
)

llm=ChatOpenAI(model="gpt-4o-mini")

first_responder_chain=first_responder_prompt | llm.bind_tools(tools=[AnswerQuestion],tool_choice="AnswerQuestion")

first_revisor_chain=first_revisor_prompt | llm.bind_tools(tools=[ReviseAnswer],tool_choice="ReviseAnswer") 


# print(response)