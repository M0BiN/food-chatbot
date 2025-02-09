
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from config import llm
from agents.generate_agent import ToGenerate
from .index import ToWebSearch


class ToGradeContent(BaseModel):
    """
    A tool for grading cleaned and structured content to evaluate if it sufficiently answers the user query.

    **Purpose:**
    - This tool is designed to assess whether the cleaned and structured content is relevant and comprehensive enough 
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
            "You are a specialized assistant tasked with evaluating the relevance of `cleaned_content` to a user's query. "
            "Your sole responsibility is to analyze the `cleaned_content` and the `user_query` and invoke the appropriate tool based on your evaluation. "
            "You must never modify the `cleaned_content` or `user_query`, respond directly to the user, or take any actions outside of invoking tools. "
            "Your entire role is limited to determining whether the `cleaned_content` sufficiently answers the `user_query` and calling the appropriate tool."
            "\n\n### Workflow Instructions:"
            "\n1. **Evaluate Relevance:**"
            "\n   - Analyze the `cleaned_content` and determine whether it sufficiently and accurately answers the `user_query`."
            "\n\n2. **Call the Appropriate Tool:**"
            "\n   - If the `cleaned_content` is relevant and sufficient to answer the `user_query`, invoke the `ToGenerate` tool with:"
            "\n     - `cleaned_content`: Pass the exact content as received without any modifications."
            "\n     - `user_query`: Pass the exact query as received without any modifications."
            "\n   - If the `cleaned_content` is irrelevant or insufficient, invoke the `ToWebSearch` tool with:"
            "\n     - `user_query`: Pass the exact query as received to guide the web search."
            "\n\n### Key Rules:"
            "\n- **No Modifications to Data:** Do not modify, rephrase, or alter the `cleaned_content` or `user_query` in any way. Use them exactly as received."
            "\n- **Tool Invocation Only:** Your role is strictly limited to invoking the `ToGenerate` or `ToWebSearch` tools based on your evaluation."
            "\n- **No User Interaction:** Do not provide any direct responses, explanations, or comments to the user."
            "\n- **Accurate Delegation:** Ensure the tools are invoked with the correct inputs, adhering strictly to their intended purpose."
        ),
        ("placeholder", "{messages}"),
    ]
)





content_grader_prompt_safe_tools = []
content_grader_prompt_sensitive_tools = []
content_grader_prompt_tools = content_grader_prompt_safe_tools + content_grader_prompt_sensitive_tools
content_grader_runnable = content_grader_prompt | llm.bind_tools(
    content_grader_prompt_tools + [ToGenerate, ToWebSearch], tool_choice="any"
)