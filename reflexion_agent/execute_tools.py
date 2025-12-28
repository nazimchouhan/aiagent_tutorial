import json
from typing import List,Dict,Any
from langchain_core.messages import AIMessage,BaseMessage,ToolMessage,HumanMessage
from langchain_community.tools import TavilySearchResults
from dotenv import load_dotenv

load_dotenv()

tavily_tool=TavilySearchResults(max_results=5,search_depth="basic")


def execute_tools(state:Dict):

    messages: List[BaseMessage] = state["messages"]

    last_ai_message = messages[-1]

    #Extract tool_calls from the AI message

    if not isinstance(last_ai_message, AIMessage):
        return {"messages": messages}

    if not getattr(last_ai_message, "tool_calls", None):
        return {"messages": messages}

    
    # Process the answerquestion or reviseanswer tool calls to extract search queries
    tool_messages=[]

    for tool_call in last_ai_message.tool_calls:
        if tool_call["name"] in ["AnswerQuestion","ReviseAnswer"]:
            call_id=tool_call["id"]
            search_queries=tool_call["args"].get("search_queries",[])

            # execute each search query using the tavily tool
            query_results={}

            for query in search_queries:
                result=tavily_tool.invoke(query)

                # create a tool message with the result
                tool_messages.append(
                    ToolMessage(
                        content=json.dumps(result),
                        tool_call_id=call_id
                    )
                )
    return {
        "messages": messages + tool_messages
    }    


test_state=[
    HumanMessage(
        content="Write about how small business can leverage AI to grow"
    ),
    AIMessage(
        content="",
        tool_calls=[
            {
                "name":"AnswerQuestion",
                "args":{
                    'answer':'',
                    'search_queries':[
                        'AI tools for small business',
                        'AI in small business marketing',
                        'AI automation or small business',
                    ],
                    'reflection':{
                        'missing':'',
                        'superfluous':'',
                    }
                },
                "id":"jdbue873edbhh837yrhd877y476",
            }
        ],
    )
]                
