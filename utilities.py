from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import ToolNode
import lancedb
from langchain_core.runnables import Runnable, RunnableConfig
from typing import Annotated, Literal, Optional
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage, RemoveMessage
from typing import Callable
from lancedb.rerankers import LinearCombinationReranker
from typing import TypedDict, List
from pydantic import ValidationError


def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
                role="tool"
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
    try:
        return [item for item in (food_table
        .search(query, query_type="hybrid")
        .limit(2)
        .rerank(reranker=reranker)
        .select(["id", "text"])
        .to_list()) if item["_relevance_score"] > min_score]
    except Exception as e:
        print(e)
        return "NO RESULT"


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

            # if self.debug:
            #     print("*"*100)
            #     print(result)
            #     print("*"*100)

            if self.is_tools_based and len(result.tool_calls)==0:
                messages = state["messages"] + [HumanMessage(content="Answer with a real output!")]
                state = {**state, "messages": messages}
                loop += 1
                if loop > 3:
                    break
            else:
                break
        return {"messages": result}
    
    def respond(self, state: dict, validator=None):
        response = []
        for attempt in range(3):
            response = self.runnable.invoke(
                {"messages": state["messages"]}, {"tags": [f"attempt:{attempt}"]}
            )
            try:
                if validator:
                    validator.invoke(response)
                return {"messages": response}
            except ValidationError as e:
                state = state + [
                    response,
                    ToolMessage(
                        content=f"{repr(e)}\n\nPay close attention to the function schema.\n\n"
                        + validator.schema_json()
                        + " Respond by fixing all validation errors.",
                        tool_call_id=response.tool_calls[0]["id"],
                    ),
                ]
        return {"messages": response}

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


def filter_last_two_tool_messages(inputs):
    """
    Extracts the last TWO ToolMessages and their corresponding AIMessages with tool_calls.
    Ensures OpenAI receives a valid tool call sequence by matching tool_call_id.
    """


    # ✅ Extract messages from ChatPromptValue
    if isinstance(inputs, dict):
        messages_list = inputs["messages"]
    else:
        messages_list = inputs.to_messages()

    last_two_pairs = []
    tool_messages_found = 0

    # 🔍 Reverse loop to find tool messages and their corresponding AI messages
    for tool_msg in reversed(messages_list):
        if isinstance(tool_msg, ToolMessage):
            tool_call_id = tool_msg.tool_call_id  # Extract tool_call_id from ToolMessage
            
            # 🔍 Find corresponding AIMessage with matching tool_calls[0]['id']
            for ai_msg in reversed(messages_list):
                if isinstance(ai_msg, AIMessage) and ai_msg.tool_calls:
                    for tool_call in ai_msg.tool_calls:
                        if tool_call['id'] == tool_call_id:  # ✅ Match tool_call_id
                            last_two_pairs.append(ai_msg)
                            last_two_pairs.append(tool_msg)
                            tool_messages_found += 1
                            break  # Stop searching for this tool message

                if tool_messages_found == 2:  # ✅ Stop after finding two valid pairs
                    break

        if tool_messages_found == 2:  # ✅ Stop after finding two valid pairs
            break

    # ✅ If no valid pairs found, return an empty list
    if not last_two_pairs:
        print("⚠️ No valid tool call sequences found.")
        return []
    return last_two_pairs  # 🔥 Contains the last two AIMessage → ToolMessage pairs



from langchain_core.messages import ToolMessage
from pydantic import BaseModel, Field
from typing import Optional

class ToSuggestionFood(BaseModel):
    """
    A tool to analyze user input and extract relevant criteria for food suggestions.
    """
    criteria: str = Field(description="The user's specific craving or food preference (e.g., spicy, fast food, Chinese).")
    context: Optional[str] = Field(description="Additional context like price range, location, or special ingredients.")





def extract_last_tool_criteria(state):
    """
    Loops over state and extracts the last AIMessage tool call with tool type 'ToSuggestionFood'.
    Returns the extracted 'criteria' and 'context'.
    """
    last_ai_message = None

    # 🔍 Reverse loop to find the last AIMessage with a tool call for 'ToSuggestionFood'
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tool_call in msg.tool_calls:
                if tool_call["name"] == "ToSuggestionFood":
                    last_ai_message = tool_call
                    break  # ✅ Stop at the first match

    # ✅ If no matching AIMessage tool call was found, return None
    if last_ai_message is None:
        print("⚠️ No AIMessage with tool call 'ToSuggestionFood' found.")
        return None

    # ✅ Parse the tool input to extract `criteria` and `context`
    tool_data = ToSuggestionFood.parse_obj(last_ai_message["args"])

    return {
        "criteria": tool_data.criteria,
        "context": tool_data.context
    }






def generate_human_message(user_intent):
    """
    Converts extracted `criteria` and `context` into a natural human request.
    Ensures the model interprets it as a real user query.
    """

    if not user_intent or not user_intent.get("criteria"):
        return HumanMessage(content="I'm looking for some food recommendations.")

    # Extract fields
    criteria = user_intent.get("criteria", "")
    context = user_intent.get("context", "")

    # Construct a natural sentence
    if criteria and context:
        message = f"I'm looking for {criteria} food options that fit this context: {context}. What do you recommend?"
    elif criteria:
        message = f"I'd like some {criteria} food recommendations. Can you help?"
    elif context:
        message = f"I'm searching for food options that match this context: {context}. Any suggestions?"
    else:
        message = "Can you suggest some good food options?"

    return HumanMessage(content=message)





def remove_unmatched_tool_messages(state):
    """
    Removes AIMessage entries with tool_calls that do not have corresponding ToolMessages.
    Ensures OpenAI API does not reject requests due to missing tool responses.

    Returns:
    - A cleaned list of messages where every AIMessage.tool_call has a matching ToolMessage.
    """
    valid_tool_call_ids = set()

    # 🔍 Step 1: Collect all valid tool_call_ids from existing ToolMessages
    for msg in state["messages"]:
        if isinstance(msg, ToolMessage) and msg.tool_call_id:
            valid_tool_call_ids.add(msg.tool_call_id)  # ✅ Store valid tool_call_id

    # 🔍 Step 2: Filter AIMessage entries that have tool_calls without valid responses
    cleaned_messages = []
    for msg in state["messages"]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            # Check if all tool_call_ids in this AIMessage have valid ToolMessages
            if any((tool_call["name"] != "CompleteOrEscalate" and tool_call["id"] not in valid_tool_call_ids) for tool_call in msg.tool_calls):
                print(msg.tool_calls)
                print(f"🚨 Removing AIMessage: tool_call_id(s) not found → {msg.tool_calls}")
                cleaned_messages.append(RemoveMessage(id=msg.id))  # ❌ Remove this message (it will be removed)

        # ✅ Keep valid AIMessage or any other message type

    return {"messages": cleaned_messages}
