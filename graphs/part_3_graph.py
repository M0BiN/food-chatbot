from langgraph.graph import StateGraph, START, END
from agents.food_search_agent import food_search_runnable, food_search_tools
from utilities import Assistant, State, create_tool_node_with_fallback
from langgraph.prebuilt import tools_condition
from tools import CompleteOrEscalate

def route_food_search(
    state: State,
):
    route = tools_condition(state)
    tool_calls = state["messages"][-1].tool_calls
    if tool_calls:
        if tool_calls[0]["name"] == CompleteOrEscalate.__name__:
            return END
    return route

builder = StateGraph(State)
builder.add_node("food_search", Assistant(food_search_runnable, debug=True))
builder.add_node("tools", create_tool_node_with_fallback(food_search_tools))
builder.add_conditional_edges("food_search", route_food_search, ["tools", END])

builder.add_edge("tools", "food_search")
builder.add_edge(START, "food_search")

part_3_graph = builder.compile()