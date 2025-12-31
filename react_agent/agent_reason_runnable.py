from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools import TavilySearchResults
from langchain.tools import tool
from langchain.agents import create_agent
load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

search_tool = TavilySearchResults(search_depth="basic", max_results=3)

@tool
def get_formatted_datetime():
    """Returns current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

tools = [search_tool, get_formatted_datetime]


agent = create_agent(
    model=llm,
    tools=tools
)
