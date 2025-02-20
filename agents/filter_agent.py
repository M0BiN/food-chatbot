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
            "📌 **Task: Strictly Clean and Structure `retrieved_content` Without Interpretation**\n"
            "You are responsible for **cleaning `retrieved_content` by filtering out unnecessary elements** while keeping the original meaning intact. "
            "Your role is **purely algorithmic**—you must NOT rephrase, summarize, interpret, or add meaning. Your only job is to **remove unwanted parts** and structure the content properly.\n\n"

            "⚠️ **DO NOT:**\n"
            "   - Modify, summarize, or rephrase the content.\n"
            "   - Add your own interpretation, meaning, or explanation.\n"
            "   - Respond to the user or perform any action beyond filtering.\n\n"

            "### 🔍 **Strict Filtering Process:**\n"
            "1️⃣ **Remove Irrelevant Elements (Without Changing Meaning):**\n"
            "   - Delete unnecessary placeholders (e.g., 'NO RESULT!'), symbols, links, advertisements, and excessive whitespace.\n"
            "   - Remove JSON fragments, broken HTML tags, tables with incomprehensible structures, or any content that is non-informative.\n"
            "   - Eliminate irrelevant metadata (page numbers, timestamps, or unrelated navigation elements).\n\n"

            "2️⃣ **Preserve Original Words and Meaning:**\n"
            "   - Keep the **exact** wording as in `retrieved_content`—do NOT summarize, interpret, or simplify.\n"
            "   - If a sentence is fragmented, do NOT attempt to infer missing information. Leave it as is.\n\n"

            "3️⃣ **Reorganize for Readability (But No Content Changes!):**\n"
            "   - Adjust spacing and sentence order **only if it improves readability** without modifying meaning.\n"
            "   - If content is scattered, remove gaps but DO NOT merge unrelated sentences.\n"
            "   - Ensure the final cleaned version is **structured and readable** but remains faithful to the original text.\n\n"

            "4️⃣ **Minimal Processing for Incomplete or Disjointed Content:**\n"
            "   - If the content is broken, leave it in its fragmented form. **Do NOT attempt to fix or fill in gaps.**\n"
            "   - If the content is too short or has no valuable information, return 'NO_RESULT'.\n\n"

            "### 🛠 **Final Step: Invoke `ToGradeContent` Tool**\n"
            "Once filtering is complete, you must call `ToGradeContent` with:\n"
            "   - `cleaned_content`: The filtered version of `retrieved_content`, with irrelevant parts removed and structure improved.\n"
            "   - `user_query`: Pass the **exact** `user_query` without modifications or assumptions.\n"
            "   - **If `retrieved_content` is empty or has no meaningful data, set `cleaned_content` to `'NO_RESULT'`.**\n\n"

            "### 🚨 **Strict Rules for Execution:**\n"
            "✔ **DO NOT change or summarize words—filter, don’t interpret.**\n"
            "✔ **DO NOT add, infer, or modify meaning—only remove unwanted elements.**\n"
            "✔ **DO NOT interact with the user—your job is to clean and invoke the tool.**\n"
            "✔ **ALWAYS invoke `ToGradeContent` with properly cleaned content—your task is incomplete until this is done.**\n\n"

            "📢 **Final Reminders:**\n"
            "   - Your job is **strictly filtering**, NOT interpreting or summarizing content.\n"
            "   - The cleaned content should be **usable and readable**, but it must **not be reworded, altered, or inferred**.\n"
            "   - Think like an **automated cleaning algorithm**—remove unnecessary junk while preserving useful content **exactly as it is written**.\n"
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