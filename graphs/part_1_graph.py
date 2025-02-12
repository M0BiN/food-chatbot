
from langgraph.prebuilt import tools_condition
from langgraph.graph import StateGraph, START, END
from utilities import create_tool_node_with_fallback
from typing import Callable
from langchain_core.messages import ToolMessage
from agents.doc_retrieval_agent import doc_retrieval_tools, doc_retrieval_runnable
from agents.filter_agent import ToFilter, filter_runnable
from agents.content_grader_agent import ToGradeContent, content_grader_runnable
from agents.web_search_agent import web_search_runnable, web_search_tools
from agents.generate_agent import ToGenerate, generate_runnable
from agents.index import ToWebSearch
from utilities import Assistant, State, create_entry_node





def route_filter(
    state: State,
):
    route = tools_condition(state)
    tool_calls = state["messages"][-1].tool_calls
    if tool_calls:
        if tool_calls[0]["name"] == ToGradeContent.__name__:
            return "content_grader"
    return route

def route_web_search(
    state: State,
):
    tool_calls = state["messages"][-1].tool_calls
    if tool_calls:
        if tool_calls[0]["name"] == ToFilter.__name__:
            return "enter_filter"
    return "web_search_tools"

def route_content_grader(
    state: State,
):
    route = tools_condition(state)
    tool_calls = state["messages"][-1].tool_calls
    if tool_calls:
        if tool_calls[0]["name"] == ToGenerate.__name__:
            return "enter_generate"
        if tool_calls[0]["name"] == ToWebSearch.__name__:
            return "enter_web_search"
    return route


def route_doc_retrieval(
    state: State,
):
    tool_calls = state["messages"][-1].tool_calls
    if tool_calls:
        if tool_calls[0]["name"] == ToFilter.__name__:
            return "enter_filter"
    return "doc_retrieval_tools"

def create_entry_node(assistant_name: str, new_dialog_state: str) -> Callable:
    def entry_node(state: State) -> dict:
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        return {
            "messages": [
                ToolMessage(
    content=f"You are now operating as {assistant_name}. Focus on your defined role and responsibilities to assist the user effectively. "
            f"Reflect on the prior conversation between the host assistant and the user to determine the appropriate course of action. "
            "The user's intent remains unresolved, and your job is to address it by utilizing the provided tools to perform the required tasks. "
            f"Remember, you are {assistant_name}"
            "Do not disclose your identity or role to the user; focus solely on fulfilling your responsibilities and ensuring the user's needs are met.",
        tool_call_id=tool_call_id,
        name=assistant_name,
        role="tool"
)

            ],
            "dialog_state": new_dialog_state,
        }

    return entry_node


builder = StateGraph(State) 

builder.add_node("filter", Assistant(filter_runnable))
builder.add_node("doc_retrieval", Assistant(doc_retrieval_runnable))
builder.add_node("content_grader", Assistant(content_grader_runnable))
builder.add_node("web_search", Assistant(web_search_runnable))
builder.add_node("generate", Assistant(generate_runnable))


builder.add_node(
    "doc_retrieval_tools", create_tool_node_with_fallback(doc_retrieval_tools)
)
builder.add_node(
    "web_search_tools", create_tool_node_with_fallback(web_search_tools)
)

builder.add_edge(START, "doc_retrieval")


builder.add_node(
    "enter_filter",
    create_entry_node("Content Filter", "filter"),
)
builder.add_edge("enter_filter", "filter")
builder.add_edge("doc_retrieval_tools", "doc_retrieval")

builder.add_conditional_edges(
    "doc_retrieval",
    route_doc_retrieval,
    [
        "enter_filter",
        "doc_retrieval_tools"
    ],
)


builder.add_node(
    "enter_content_grader",
    create_entry_node("Content Grader", "content_grader"),
)
builder.add_edge(
    "filter",
    "enter_content_grader"
)
builder.add_edge("enter_content_grader", "content_grader")


builder.add_conditional_edges(
    "content_grader",
    route_content_grader,
    [
        "enter_generate",
        "enter_web_search"
    ],
)
builder.add_conditional_edges(
    "web_search",
    route_web_search,
    [
        "enter_filter",
        "web_search_tools"
    ],
)

builder.add_node(
    "enter_web_search",
    create_entry_node("Web Search Assistant", "web_search"),
)
builder.add_edge("enter_web_search", "web_search")

builder.add_edge("web_search_tools", "web_search")
builder.add_node(
    "enter_generate",
    create_entry_node("Answer Generate Assistant", "generate"),
)
builder.add_edge("enter_generate", "generate")
builder.add_edge("generate", END)



part_1_graph = builder.compile()


