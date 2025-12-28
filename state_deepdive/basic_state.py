from typing import TypedDict
from langgraph.graph import END,StateGraph
class Graphstate(TypedDict):
    count:int


def increment(state:Graphstate)->Graphstate:

    return {
        "count":state["count"]+1
    }

def should_continue(state:Graphstate):

    if(state["count"]<10):
        return "increment"
    return END


graph=StateGraph(Graphstate)

graph.add_node("increment",increment)
graph.add_conditional_edges("increment",should_continue)
graph.set_entry_point("increment")

app=graph.compile()

print(app.get_graph().draw_mermaid())

app.get_graph().print_ascii()

state={
    "count":0
}
result=app.invoke(state)

print(result)
