from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage
from config import llm

# Define messages (mix of human, tool, and AI messages)
messages = [
    HumanMessage(content="What spicy dishes are available?"),
    AIMessage(content="Let me find some spicy dishes for you.", tool_calls=[{"id": "tool_1234", "name": "GetFoodOptions", "args": {}}]),
    ToolMessage(content="Spicy Ramen is available for $12.", tool_call_id="tool_1234"),
    ToolMessage(content="Chili Tacos are available for $10.", tool_call_id="tool_1234")
]

# Define the ChatPromptTemplate - Using a placeholder for messages
food_tool_prompt = ChatPromptTemplate.from_messages([
    ("system", "🔧 You are processing only tool messages."),
    MessagesPlaceholder(variable_name="messages")
])

from langchain_core.messages import ToolMessage, AIMessage

from langchain_core.messages import ToolMessage, AIMessage

from langchain_core.messages import ToolMessage, AIMessage

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



# 🔥 Modify the pipeline to include filtering
food_suggestion_runnable = (
    food_tool_prompt 
    | filter_last_two_tool_messages  # Filter messages dynamically
    | llm
)

# Running the pipeline (it will now only process ToolMessages)
response = food_suggestion_runnable.invoke({"messages": messages})
print(response)
