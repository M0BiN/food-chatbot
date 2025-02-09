from langgraph.graph import StateGraph, START, END
from agents.order_management_agent import order_management_runnable, order_management_safe_tools, order_management_sensitive_tools
from utilities import Assistant, State, create_tool_node_with_fallback
from langgraph.prebuilt import tools_condition
from tools import CompleteOrEscalate

def route_order_management(
    state: State,
):
    route = tools_condition(state)
    tool_calls = state["messages"][-1].tool_calls
    if tool_calls:
        if tool_calls[0]["name"] == CompleteOrEscalate.__name__:
            return END
        if tool_calls[0]["name"] in [func.name for func in order_management_sensitive_tools]:
            return "sensitive_tools"
    return route

builder = StateGraph(State)
builder.add_node("order_management", Assistant(order_management_runnable, debug=True))
builder.add_node("tools", create_tool_node_with_fallback(order_management_safe_tools))
builder.add_node("sensitive_tools", create_tool_node_with_fallback(order_management_sensitive_tools))

builder.add_conditional_edges("order_management", route_order_management, ["tools", "sensitive_tools", END])

builder.add_edge("tools", "order_management")
builder.add_edge("sensitive_tools", "order_management")

builder.add_edge(START, "order_management")

part_2_graph = builder.compile(interrupt_before=["sensitive_tools"])