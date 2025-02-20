
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from config import llm
from agents.generate_agent import ToGenerate
from langchain_core.messages import ToolMessage, AIMessage
from .index import ToWebSearch


examples = [
    # Example 1: Cleaned content is sufficient → Use `ToGenerate`
    AIMessage(
        "",
        name="example_assistant",
        tool_calls=[
            {
                "name": "ToGradeContent",
                "args": {
                    "cleaned_content": "Adas Polo is a Persian dish made with rice, lentils, and caramelized onions.",
                    "user_query": "What is Adas Polo?"
                },
                "id": "1"
            }
        ],
    ),
    ToolMessage(
        content="You are now operating as Content Grader. Focus on your defined role and responsibilities to assist the user effectively. "
                "Reflect on the prior conversation between the host assistant and the user to determine the appropriate course of action. "
                "The user's intent remains unresolved, and your job is to address it by utilizing the provided tools to perform the required tasks. "
                "Remember, you are Content Grader. "
                "Do not disclose your identity or role to the user; focus solely on fulfilling your responsibilities and ensuring the user's needs are met. "
                "Sufficient",
        tool_call_id="1",
        name="Content Grader",
        role="tool"
    ),
    AIMessage(
        "",
        name="example_assistant",
        tool_calls=[
            {
                "name": "ToGenerate",
                "args": {
                    "cleaned_content": "Adas Polo is a Persian dish made with rice, lentils, and caramelized onions.",
                    "user_query": "What is Adas Polo?"
                },
                "id": "2"
            }
        ],
    ),
    ToolMessage(
        content="",
        tool_call_id="2",
        name="Generated",
        role="tool"
    ),
    # Example 2: Cleaned content lacks key details → Requires `ToWebSearch`
    AIMessage(
        "",
        name="example_assistant",
        tool_calls=[
            {
                "name": "ToGradeContent",
                "args": {
                    "cleaned_content": "Tahini is a paste made from sesame seeds.",
                    "user_query": "What are the health benefits of tahini?"
                },
                "id": "3"
            }
        ],
    ),
    ToolMessage(
        content="You are now operating as Content Grader. Focus on your defined role and responsibilities to assist the user effectively. "
                "Reflect on the prior conversation between the host assistant and the user to determine the appropriate course of action. "
                "The user's intent remains unresolved, and your job is to address it by utilizing the provided tools to perform the required tasks. "
                "Remember, you are Content Grader. "
                "Do not disclose your identity or role to the user; focus solely on fulfilling your responsibilities and ensuring the user's needs are met. "
                "Insufficient",
        tool_call_id="3",
        name="Content Grader",
        role="tool"
    ),
    AIMessage(
        "",
        name="example_assistant",
        tool_calls=[
            {
                "name": "ToWebSearch",
                "args": {
                    "user_query": "What are the health benefits of tahini?"
                },
                "id": "4"
            }
        ],
    ),
    ToolMessage(
        content="",
        tool_call_id="4",
        name="Searching",
        role="tool"
    ),
    # Example 3: Cleaned content is relevant and sufficient → Use `ToGenerate`
    AIMessage(
        "",
        name="example_assistant",
        tool_calls=[
            {
                "name": "ToGradeContent",
                "args": {
                    "cleaned_content": "Fesenjan is a Persian stew made with pomegranate molasses, ground walnuts, and either chicken or duck.",
                    "user_query": "How is Fesenjan made?"
                },
                "id": "5"
            }
        ],
    ),
    ToolMessage(
        content="You are now operating as Content Grader. Focus on your defined role and responsibilities to assist the user effectively. "
                "Reflect on the prior conversation between the host assistant and the user to determine the appropriate course of action. "
                "The user's intent remains unresolved, and your job is to address it by utilizing the provided tools to perform the required tasks. "
                "Remember, you are Content Grader. "
                "Do not disclose your identity or role to the user; focus solely on fulfilling your responsibilities and ensuring the user's needs are met. "
                "Sufficient",
        tool_call_id="5",
        name="Content Grader",
        role="tool"
    ),
    AIMessage(
        "",
        name="example_assistant",
        tool_calls=[
            {
                "name": "ToGenerate",
                "args": {
                    "cleaned_content": "Fesenjan is a Persian stew made with pomegranate molasses, ground walnuts, and either chicken or duck.",
                    "user_query": "How is Fesenjan made?"
                },
                "id": "6"
            }
        ],
    ),
    ToolMessage(
        content="",
        tool_call_id="6",
        name="Generated",
        role="tool"
    ),
    # Example 4: Cleaned content contains similar words but lacks a full answer → Use `ToWebSearch`
    AIMessage(
        "",
        name="example_assistant",
        tool_calls=[
            {
                "name": "ToGradeContent",
                "args": {
                    "cleaned_content": "Saffron is often used in Persian cuisine for its flavor and color.",
                    "user_query": "What are the health benefits of saffron?"
                },
                "id": "7"
            }
        ],
    ),
    ToolMessage(
        content="You are now operating as Content Grader. Focus on your defined role and responsibilities to assist the user effectively. "
                "Reflect on the prior conversation between the host assistant and the user to determine the appropriate course of action. "
                "The user's intent remains unresolved, and your job is to address it by utilizing the provided tools to perform the required tasks. "
                "Remember, you are Content Grader. "
                "Do not disclose your identity or role to the user; focus solely on fulfilling your responsibilities and ensuring the user's needs are met. "
                "Insufficient",
        tool_call_id="7",
        name="Content Grader",
        role="tool"
    ),
    AIMessage(
        "",
        name="example_assistant",
        tool_calls=[
            {
                "name": "ToWebSearch",
                "args": {
                    "user_query": "What are the health benefits of saffron?"
                },
                "id": "8"
            }
        ],
    ),
        ToolMessage(
        content="",
        tool_call_id="8",
        name="Searching",
        role="tool"
    ),
]




class ToGradeContent(BaseModel):
    """
    A tool for grading cleaned and structured content to evaluate if it sufficiently answers the user query.

    **Purpose:**
    - Designed to assess whether the cleaned and structured content is relevant and comprehensive enough 
      to address the user's original query.
    - It ensures that the content can serve as a reliable basis for answering the query.

    **Usage Guidelines:**
    - Use this tool after the content has been cleaned and structured.
    - Populate the `cleaned_content` field with the refined text that needs to be evaluated.
    - Provide the `user_query` to establish the context and criteria for grading the content.
    - The output should indicate whether the content can sufficiently answer the query or if further refinement or retrieval is needed.
    """

    cleaned_content: str = Field(
        description="The cleaned and structured content to be graded for its relevance and adequacy in answering the user query."
    )
    user_query: str = Field(
        description="The original user query that provides the context and criteria for grading the content."
    )




# Retrieval Grader assistant

content_grader_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "📌 **Task Overview:**\n"
            "You are responsible for evaluating whether `cleaned_content` contains **enough relevant information** to generate a valid answer to `user_query`. "
            "Your role is to **avoid unnecessary rejections and web searches** while ensuring that the response remains meaningful and complete.\n\n"

            "⚠️ **What You MUST NOT Do:**\n"
            "   - **Do NOT assume `cleaned_content` is incomplete just because minor details are missing.**\n"
            "   - **Do NOT default to `ToWebSearch` unless there is a critical gap in information.**\n"
            "   - **Do NOT rely on word similarity to decide sufficiency—assess if the query is actually answered.**\n"
            "   - **Do NOT modify or alter `cleaned_content` or `user_query`.**\n"
            "   - **Do NOT respond directly to the user—your task is to evaluate and invoke tools.**\n\n"

            "### 🔍 **How to Determine Sufficiency:**\n"
            "1️⃣ **Evaluate If `cleaned_content` Provides a Meaningful Answer:**\n"
            "   - If the content **logically addresses the user query**, it is sufficient.\n"
            "   - **Food-related queries do NOT require exhaustive details**—a dish's name, ingredients, and preparation can be enough.\n"
            "   - If `cleaned_content` is **clear, relevant, and can give insight to address the query**, it is sufficient.\n"
            "   - If **critical details are missing** (e.g., the query asks for ingredients but none are listed), then it is insufficient.\n\n"

            "2️⃣ **Select the Correct Tool Based on Sufficiency:**\n"
            "   - ✅ **If `cleaned_content` provides enough relevant information to form an answer, invoke `ToGenerate` with:**\n"
            "     - `cleaned_content`: Pass it exactly as received (unchanged).\n"
            "     - `user_query`: Pass it exactly as received (unchanged).\n\n"
            "   - ❌ **Only if `cleaned_content` is missing essential details and cannot answer the query, invoke `ToWebSearch` with:**\n"
            "     - `user_query`: Pass it exactly as received (unchanged).\n\n"

            "### 🚨 **Strict Rules to Follow:**\n"
            "✔ **Do NOT reject `cleaned_content` just because it lacks extra, non-essential details.**\n"
            "✔ **Do NOT accept content just because it contains similar words—verify if it actually answers the question.**\n"
            "✔ **For food-related queries, consider if `cleaned_content` gives key information (e.g., ingredients, preparation, or cultural relevance).**\n"
            "✔ **If the content allows for a valid response to be generated, use `ToGenerate`. Do NOT call `ToWebSearch` unless truly necessary.**\n"
            "✔ **Your goal is to prevent redundancy—if an answer can be formed, do NOT trigger an unnecessary search.**\n"
            "✔ **Do NOT alter the content—only assess whether it can sufficiently answer the query.**\n\n"

            "📢 **Final Reminders:**\n"
            "   - `cleaned_content` **does NOT need to cover every possible aspect of `user_query`**, just what is necessary for a valid response.\n"
            "   - If a dish’s name is given, an explanation of its **ingredients and preparation** may be sufficient—extra history or trivia are optional.\n"
            "   - If `cleaned_content` contains **enough factual information to construct a reasonable response**, prefer `ToGenerate`.\n"
            "   - **Only call `ToWebSearch` if a critical piece of information is missing and prevents a valid response.**\n"
            "   - **Your role is to ensure efficiency while maintaining accuracy**—minimize redundant web searches.\n"
        ),
        *examples,
        ("placeholder", "{messages}"),
    ]

)







content_grader_prompt_safe_tools = []
content_grader_prompt_sensitive_tools = []
content_grader_prompt_tools = content_grader_prompt_safe_tools + content_grader_prompt_sensitive_tools
content_grader_runnable = content_grader_prompt | llm.bind_tools(
    content_grader_prompt_tools + [ToGenerate, ToWebSearch], tool_choice="any"
)