from typing import TypedDict,List
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END,StateGraph
from chain import generation_chain,reflex_chain
from langchain_core.messages import AIMessage

load_dotenv()

class GraphState(TypedDict):
    messages: List[BaseMessage]


graph=StateGraph(GraphState)

REFLECT="reflect"
GENERATE="generate"

def generate_node(state: GraphState):
    response= generation_chain.invoke({
        "name": "ViralTweetBot",
        "messages":state["messages"],
        "user_input": state["messages"][-1].content
    })
    return {"messages": state["messages"] + [AIMessage(content=response.content)]}

def reflect_node(state:GraphState):
    response= reflex_chain.invoke({
        "name": "ReflexTweetBot",
        "user_input": state["messages"][-1].content
    })
    return {"messages": state["messages"] + [AIMessage(content=response.content)]}

graph.add_node(GENERATE,generate_node)
graph.add_node(REFLECT,reflect_node)

graph.set_entry_point(GENERATE)

def should_continue(state:GraphState):
    if(len(state['messages'])>4):
        return END
    return REFLECT

graph.add_conditional_edges(GENERATE,should_continue,
    {
        REFLECT: REFLECT,
        END: END
    }
)

graph.add_edge(REFLECT,GENERATE)

app=graph.compile()

print(app.get_graph().draw_mermaid())

app.get_graph().print_ascii()


result=app.invoke({
    "messages":[
        HumanMessage(content="write a viral tweet about Indian mens cricket team")
    ]
})

print(result["messages"][-1].content)
