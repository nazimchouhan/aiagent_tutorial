from typing import List,TypedDict
from langchain_core.messages import BaseMessage,ToolMessage,HumanMessage,AIMessage
from langgraph.graph import END,StateGraph
from chains import first_responder_chain,first_revisor_chain
from execute_tools import execute_tools

class GraphState(TypedDict):
    messages: List[BaseMessage]
    iteration: int

MAX_ITERATIONS=2
graph=StateGraph(GraphState)

def responder_node(state: GraphState):
    result = first_responder_chain.invoke({
        "messages": state["messages"]
    })
    # result is AIMessage with tool_calls
    return {
        "messages": state["messages"] + [result],
        "iteration": state.get("iteration", 0)
    }

def revisor_node(state: GraphState):
    # Filter out both ToolMessages AND AIMessages with tool_calls
    clean_messages = []
    for m in state["messages"]:
        # Skip ToolMessages
        if isinstance(m, ToolMessage):
            continue
        # Skip AIMessages that have tool_calls
        if isinstance(m, AIMessage) and hasattr(m, 'tool_calls') and m.tool_calls:
            continue
        clean_messages.append(m)
    
    result = first_revisor_chain.invoke({
        "messages": clean_messages
    })
    current_iteration = state.get("iteration", 0) + 1

    # result is AIMessage with tool_calls
    return {
        "messages": state["messages"] + [result],
        "iteration": current_iteration
    }


graph.add_node("responder",responder_node)
graph.add_node("execute_tools",execute_tools)

graph.add_node("revisor",revisor_node)

graph.add_edge("responder","execute_tools")
graph.add_edge("execute_tools","revisor")


def event_loop(state:GraphState):
    current_iteration = state.get("iteration", 0)
    print(f"EVENT LOOP → iteration: {current_iteration}/{MAX_ITERATIONS}")
    
    if current_iteration >= MAX_ITERATIONS:
        print("ENDING GRAPH - Max iterations reached")
        return END
    
    print("LOOPING BACK TO execute_tools")
    return "execute_tools"

graph.add_conditional_edges("revisor",event_loop,
    {
        "execute_tools": "execute_tools",
        END: END
    }
)

graph.set_entry_point("responder")

app=graph.compile()

print(app.get_graph().draw_mermaid())

app.get_graph().print_ascii()

result = app.invoke({
    "messages": [
        HumanMessage(content="Write a blog post about discipline")
    ]
})

print("\nFINAL OUTPUT:\n")
print(result["messages"][-1].content)
final_messages = result["messages"]
for msg in reversed(final_messages):
    if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
        for tool_call in msg.tool_calls:
            if tool_call["name"] == "ReviseAnswer":
                answer = tool_call["args"].get("answer", "")
                references = tool_call["args"].get("references", [])
                print(answer)
                if references:
                    print("\n\nReferences:")
                    for ref in references:
                        print(f"- {ref}")
                break
        break

