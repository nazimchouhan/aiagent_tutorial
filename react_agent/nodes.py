from langchain_core.agents import AgentAction, AgentFinish
from react_state import AgentState
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
    BaseMessage,
)
from agent_reason_runnable import agent, tools

# ---------------- Reason Node ----------------
def reason_node(state: AgentState):
    """
    LLM reasoning step
    """

    # Convert state into chat messages
    messages = [HumanMessage(content=state["input"])]

    # Invoke agent correctly
    result = agent.invoke({
        "messages": messages
    })

    return {
        "agent_outcome": result
    }

# ---------------- Act Node ----------------
def Act_node(state: AgentState):
    """
    Executes tool selected by the agent
    """

    action = state["agent_outcome"]

    if not isinstance(action, AgentAction):
        return {}

    tool_name = action.tool
    tool_input = action.tool_input

    tool_fn = next((t for t in tools if t.name == tool_name), None)

    if tool_fn is None:
        observation = f"Tool '{tool_name}' not found"
    else:
        try:
            if isinstance(tool_input, dict):
                observation = tool_fn.invoke(tool_input)
            else:
                observation = tool_fn.invoke(tool_input)
        except Exception as e:
            observation = f"Tool error: {e}"

    return {
        "intermediate_steps": [(action, str(observation))]
    }
