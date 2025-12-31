from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.agents import AgentFinish
from react_state import AgentState
from nodes import reason_node, Act_node

load_dotenv()

REASON = "reason"
ACT = "act"

def should_continue(state: AgentState):
    if isinstance(state["agent_outcome"], AgentFinish):
        return END
    return ACT


graph = StateGraph(AgentState)

graph.add_node(REASON, reason_node)
graph.add_node(ACT, Act_node)

graph.set_entry_point(REASON)

graph.add_conditional_edges(REASON, should_continue)
graph.add_edge(ACT, REASON)

app = graph.compile()

result = app.invoke({
    "input": "How many days ago was the latest SpaceX launch?",
    "agent_outcome": None,
    "intermediate_steps": []
})

print("\n=== FINAL ANSWER ===")
final = result["agent_outcome"]
if isinstance(final, AgentFinish):
    print(final.return_values["output"])
else:
    print("Agent did not finish properly")



