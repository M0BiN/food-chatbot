from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from config import llm
from agents.content_grader_agent import ToGradeContent



class ToFilter(BaseModel):
    """
    A tool for cleaning and refining plain, raw, and unstructured retrieved data.

    **Purpose:**
    - This tool is specifically designed to transform raw, messy, and unstructured text into clean, concise, 
      and well-structured content that is easy to read and understand.
    - It ensures that only essential and valuable information is retained while removing unnecessary or irrelevant elements.

    **Usage Guidelines:**
    - **Always use this tool whenever you have retrieved data**.
    - Populate the `retrieved_content` field with the raw data that requires cleaning.
    - Provide the original query in the `user_query` field to help contextualize and guide the refinement process.
    - Use this tool to enhance the clarity and usability of the retrieved content, making it more structured and valuable for users.
    """

    retrieved_content: str = Field(
        description="The raw, plain content retrieved from the source."
    )
    user_query: str = Field(
        description="The original user query that guided the retrieval process, used to provide context for refining the content."
    )

# filter assistant

filter_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Your task is to clean the `retrieved_content` by removing unnecessary words, odd characters, or irrelevant elements, "
            "and organizing the remaining text into clear, readable sentences or paragraphs. Your job is strictly to process the raw content "
            "and prepare it for evaluation. You must never respond to the user or perform any actions beyond cleaning and invoking the tool."
            "\n\n### Rules for Cleaning:"
            "\n1. **Remove Unnecessary Elements:**"
            "\n   - Eliminate placeholders (e.g., 'NO RESULT!'), symbols, links, excessive spaces, JSON fragments, tables, or irrelevant characters."
            "\n   - Remove anything that does not contribute to the clarity or structure of the content."
            "\n2. **Do Not Change Original Words:**"
            "\n   - Keep the original words intact. Do not rephrase, summarize, or alter the meaning of the content in any way."
            "\n3. **Reorganize Sentences:**"
            "\n   - Rearrange sentences or lines to create a logical, readable structure, but without adding, interpreting, or modifying the content."
            "\n4. **Minimal Processing for Fragmented Content:**"
            "\n   - If the content is disjointed or incomplete, clean it minimally without interpreting or connecting unrelated topics."
            "\n\n### Final Step: Invoke the Tool"
            "\nOnce the content is cleaned, invoke the `ToGradeContent` tool with the following inputs:"
            "\n   - `cleaned_content`: The version of the `retrieved_content` with unnecessary elements removed and sentences reorganized."
            "\n   - `user_query`: Pass the query as it is, without modifying or connecting it to the cleaned content."
            "\n\n### Strict Rules:"
            "- **NEVER Respond to the User:** Your role is strictly to clean and structure the content. Do not interact with the user or explain your actions."
            "- **No Interpretation or Analysis:** Do not analyze or interpret the content. Focus only on removing unnecessary parts and organizing sentences."
            "- **Preserve Original Meaning:** Ensure that the original intent and meaning remain unchanged in the cleaned content."
            "- **Tool Invocation Only:** Your job is not complete until you invoke the `ToGradeContent` tool with properly cleaned content."
            "\n\n### Key to Success:"
            "Your success depends on how well you remove unnecessary elements and reorganize sentences without altering the original meaning. "
            "Always ensure the `ToGradeContent` tool is called correctly and avoid responding to the user."
            "even if there is no 'retrieved_content', just call the `ToGradeContent` with the 'NO_RESULT' as cleaned_content"
        ),
        ("placeholder", "{messages}"),
    ]
)








filter_safe_tools = []
filter_sensitive_tools = []
filter_tools = filter_safe_tools + filter_sensitive_tools
filter_runnable = filter_prompt | llm.bind_tools(
    filter_tools + [ToGradeContent], tool_choice="any"
)