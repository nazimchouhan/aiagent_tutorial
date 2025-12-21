from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

llm=ChatOpenAI(model="nvidia/nemotron-3-nano-30b-a3b:free")

search_tool=TavilySearchResults(search_depth="basic")

tools=[search_tool]
agent=create_react_agent(llm,tools)


result = agent.invoke({
    "messages": [
        ("user", "Give me info about today's Delhi weather")
    ]
})

print(result["messages"][-1].content)