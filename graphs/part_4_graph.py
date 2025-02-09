from langgraph.graph import StateGraph, START, END
from agents.food_suggestion_agent import food_suggestion_runnable, food_suggestion_tools
from utilities import Assistant, State, create_tool_node_with_fallback
from langgraph.prebuilt import tools_condition

def route_food_suggestion(
    state: State,
):
    route = tools_condition(state)
    tool_calls = state["messages"][-1].tool_calls
    # if tool_calls:
    #     if tool_calls[0]["name"] == CompleteOrEscalate.__name__:
    #         return END
    return route

builder = StateGraph(State)
builder.add_node("food_suggestion", Assistant(food_suggestion_runnable, debug=True))
builder.add_node("tools", create_tool_node_with_fallback(food_suggestion_tools))
builder.add_conditional_edges("food_suggestion", route_food_suggestion, ["tools", END])

builder.add_edge("tools", "food_suggestion")
builder.add_edge(START, "food_suggestion")

part_4_graph = builder.compile()