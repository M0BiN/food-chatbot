from langchain_core.prompts import ChatPromptTemplate
from config import llm
from langchain_core.messages import HumanMessage, RemoveMessage


# Summarize Conversation Prompt
summarize_conversation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "📝 **Conversation Summarization Expert**\n\n"
            "Your task is to create a **concise and informative summary** of the ongoing conversation. "
            "You must ensure that all key details remain while removing unnecessary or repetitive information. "
            "This summary will be used for tracking important user requests and context in a **brief yet effective manner**.\n\n"

            "### 🔹 **Guidelines for Summarization:**\n"
            "1️⃣ **Maintain Key Information**\n"
            "- Preserve user preferences, requests, and constraints.\n"
            "- Retain any important clarifications or decisions made.\n"
            "- Ensure that the summary stays relevant and does not drift off-topic.\n\n"

            "2️⃣ **Summarize Efficiently**\n"
            "- Keep the summary **short and to the point** while ensuring clarity.\n"
            "- Remove filler words, small talk, and redundant statements.\n"
            "- Avoid over-explaining—**capture intent, not every word.**\n\n"

            "3️⃣ **Update & Improve the Summary**\n"
            "- If a previous summary exists, **extend and refine** it with new details.\n"
            "- Merge new information smoothly rather than repeating old content.\n"
            "- Keep the summary **organized and easy to understand**.\n\n"

            "### ❗ **Final Rules**\n"
            "✅ **Ensure all user requests and preferences are captured.**\n"
            "✅ **Make the summary concise, structured, and easy to read.**\n"
            "🚫 **Do not include unnecessary details or repeat past messages.**\n"
            "🚫 **Do not speculate—only summarize what was actually said.**\n\n"

            "Your role is to **maintain an up-to-date, relevant summary of the conversation** "
            "so that important details remain available while keeping it brief and readable."
        ),
        ("user", "{messages}")
    ]
)








summarize_conversation_safe_tools = []
summarize_conversation_sensitive_tools = []
summarize_conversation_tools = summarize_conversation_safe_tools + summarize_conversation_sensitive_tools

summarize_conversation_runnable = summarize_conversation_prompt | llm.bind_tools(
    summarize_conversation_tools + []
)



def summarize_conversation(state):
    
    # First, we get any existing summary
    summary = state.get("summary", "")

    # Create our summarization prompt 
    if summary:
        
        # A summary already exists
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
        
    else:
        summary_message = "Create a summary of the conversation above:"

    # Add prompt to our history
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = summarize_conversation_runnable.invoke(messages)

    # Delete all but the 3 most recent messages
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-3]]
    return {"summary": response.content, "messages": delete_messages}


def should_summarize(state):
    
    """Return the next node to execute."""
    
    messages = state["messages"]
    # print("&&"*100)
    # print(len(messages))
    # If there are more than 18 messages, then we summarize the conversation
    if len(messages) > 18:
        return "summarize_conversation"
    
    # Otherwise we can just end
    return "primary_assistant"