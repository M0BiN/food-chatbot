from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import ToolNode
import lancedb
from langchain_core.runnables import Runnable, RunnableConfig
from typing import Annotated, Literal, Optional
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.messages import ToolMessage, HumanMessage
from typing import Callable
from lancedb.rerankers import LinearCombinationReranker
from typing import TypedDict, List


def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


def _print_event(events: dict, _printed: set, max_length=1500):
    for message in events.get("messages", []):
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)
    return

    # if current_state:
    #     print("Currently in: ", current_state[-1])
    for event in events:
        message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)



reranker = LinearCombinationReranker()
db = lancedb.connect('./lancedb')
food_table = db.open_table("food")


class SearchResult(TypedDict):
    id: int
    text: str
    _relevance_score: float

def document_search(query:str, min_score:float=0.65)->List[SearchResult]:
    return [item for item in (food_table
    .search(query, query_type="hybrid")
    .limit(2)
    .rerank(reranker=reranker)
    .select(["id", "text"])
    .to_list()) if item["_relevance_score"] > min_score]


def update_dialog_stack(left: list[str], right: Optional[str]) -> list[str]:
    """Push or pop the state."""
    
    return ["primary_assistant"]
    if right is None:
        return left
    if right == "pop":
        return left[:-1]
    return left + [right]


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: str
    summary: str = ""
    dialog_state: Annotated[
        list[
            Literal[
    "primary_assistant",
    "fetch_user_info",
]
        ],
        update_dialog_stack,
    ] = "fetch_user_info"


class Assistant:
    def __init__(self, runnable: Runnable, is_tools_based:bool=False, debug:bool=False):
        self.runnable = runnable
        self.is_tools_based = is_tools_based
        self.debug = debug
    def __call__(self, state: State, config: RunnableConfig):
        loop = 0
        while True:
            configuration = config.get("configurable", {})
            user_info = configuration.get("user_info", None)
            summary = configuration.get("summary", None)
            state = {**state, "user_info": user_info, "summary":summary}

            result = self.runnable.invoke(state)
            print("*"*100)
            print(result)
            print("*"*100)
            # if self.debug:
            #     print("*"*100)
            #     print(result)
            #     print("*"*100)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.

            if self.is_tools_based and len(result.tool_calls)==0:
                messages = state["messages"] + [HumanMessage(content="Answer with a real output!")]
                state = {**state, "messages": messages}
                loop += 1
                if loop > 3:
                    break
            else:
                break
        return {"messages": result}

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
    name=assistant_name
)

            ],
            "dialog_state": new_dialog_state,
        }
    

    return entry_node


