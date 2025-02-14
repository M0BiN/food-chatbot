from langgraph.graph import StateGraph, START, END
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from utilities import extract_last_tool_criteria, generate_human_message, filter_last_two_tool_messages, remove_unmatched_tool_messages
from langchain_core.tools import StructuredTool
from tools import available_food_search, CompleteOrEscalate
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, HumanMessage

from agents.reflextions_agents import (
    food_suggestion_runnable,
    food_revision_runnable,
    ReviseFoodRecommendation, FoodRecommendation)
from utilities import Assistant, State

search = TavilySearchAPIWrapper()
tavily_tool = TavilySearchResults(api_wrapper=search, max_results=5)






def run_queries(search_queries: list[str], **kwargs):
    """Run the generated queries."""
    return tavily_tool.batch([{"query": query} for query in search_queries])


tool_node = ToolNode(
    [
        StructuredTool.from_function(run_queries, name=FoodRecommendation.__name__),
        StructuredTool.from_function(run_queries, name=ReviseFoodRecommendation.__name__),
        available_food_search,
        CompleteOrEscalate
    ]
)


def _get_num_iterations(state: list):
    i = 0
    for msg in reversed(state):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tool_call in msg.tool_calls:
                if tool_call["name"] == "ReviseFoodRecommendation":
                    i += 1
        elif isinstance(msg, HumanMessage):
            break
    return i


MAX_ITERATIONS = 2
def route_food_suggestion(
    state: State,
):
    if hasattr(state["messages"][-1], "tool_calls") and state["messages"][-1].tool_calls:
        tool_calls = state["messages"][-1].tool_calls
    else:
        tool_calls = None
    if tool_calls and tool_calls[0]["name"] == CompleteOrEscalate.__name__:
        return END
    # in our case, we'll just stop after N plans

    return "execute_tools"

def draft_node(state: State):
    first_responder = Assistant(food_suggestion_runnable)
    
    # ✅ Extract user intent (criteria & context)
    user_intent = extract_last_tool_criteria(state)

    # ✅ Generate a proper human-like message
    human_message = generate_human_message(user_intent)
    
    # ✅ Ensure the response is a valid list of messages
    response_messages = first_responder.respond({"messages": [human_message]})

    # ✅ Fix AI messages that have empty content
    for msg in response_messages:
        if isinstance(msg, AIMessage) and not msg.content:
            msg.content = "Executing tool call..."  # 🔥 Add default content to prevent LangChain error
    return {"messages": [response_messages["messages"]]}


def revisor_node(state: State):
    num_iterations = _get_num_iterations(state["messages"])
    if num_iterations > MAX_ITERATIONS:
        print("@"*110)
        return {"messages":[AIMessage(
        content="",
        tool_calls=[
            {
                "name": "CompleteOrEscalate",
                "id": "fake_call_CompleteOrEscalate",
                "args": {
                    "cancel": False,
                    "reason": "Task was successfully completed, no escalation needed."
                },
                "type": "tool_call"
            }
        ]
    )]}
    user_intent = extract_last_tool_criteria(state)
    revisor = Assistant(food_revision_runnable)
    response_messages = revisor.respond(
        {
            "messages": [
                generate_human_message(user_intent),
            ] + filter_last_two_tool_messages(state)
        }
    )


    return {"messages": response_messages["messages"]}


builder = StateGraph(State)
builder.add_node("draft", draft_node)
builder.add_node("revisor", revisor_node)
builder.add_node("execute_tools", tool_node)
builder.add_edge("__start__", "draft")

builder.add_edge("draft", "execute_tools")

builder.add_edge("execute_tools", "revisor")
builder.add_conditional_edges("revisor", route_food_suggestion, [END, "execute_tools"])
builder.add_edge(START, "draft")

part_5_graph = builder.compile()
