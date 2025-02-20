
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from config import llm
from tools import retrieve_from_doc
from agents.filter_agent import ToFilter
from langchain_core.tools import tool

@tool
class ToDocRetrieval(BaseModel):
    """
    A tool for delegating food-related queries to a specialized assistant.

    **Purpose:**
    - Use this tool to transfer user queries requiring expertise in food-related topics 
      to a specialized assistant who is proficient in answering such questions.

    **Usage:**
    - Populate the `user_query` field with the user's original question.
    - Ensure the query is clearly related to food to make the delegation effective.

    **Field Descriptions:**
    - `user_query`: A string containing the user's original question that needs to be 
      handled by a food-related query expert.
    """

    user_query: str = Field(
        description="The user's original query requiring expertise in food-related topics."
    )

    class Config:
        json_schema_extra = {
            "example1": {
                "user_query": "What are the health benefits of olive oil?",
            },
            "example2": {
                "user_query": "can i drink tea right after my meal?",
            }
        }


# Doc Retrieval assistant

doc_retrieval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "📌 **Task Overview:**\n"
            "You are a specialized assistant responsible for **retrieving food-related information from documents**. "
            "Your primary role is to **optimize the search query for the document database** to maximize retrieval accuracy.\n\n"

            "🚀 **Your Goal:**\n"
            "   - **Find the most relevant information** by improving the search query when necessary.\n"
            "   - **Ensure retrieved content is clean and structured** using the `ToFilter` tool.\n"
            "   - **Maintain strict accuracy**—do not add or remove meaning from the retrieved content.\n\n"

            "### 🔍 **Step 1: Optimize the Search Query for `retrieve_from_doc`**\n"
            "📌 **Your most important task is to adjust the user query so that it is more likely to retrieve useful results.**\n"
            "   - **DO NOT search for the exact user query if it is too broad, specific, or unlikely to exist in the documents.**\n"
            "   - Instead, **generalize, restructure, or expand the query** while keeping the user’s intent intact.\n"
            "   - Ensure the query is **concise, specific, and focuses on keywords that are more likely to be present in the document database.**\n\n"

            "### 🔄 **Examples of Search Query Adjustments**\n"
            "❌ **Bad Query:** 'What is pasta?' (This exact question is unlikely to be found in documents.)\n"
            "✅ **Better Query:** 'Cooking pasta techniques' (More relevant and likely to match document content.)\n\n"
            "❌ **Bad Query:** 'Are tomatoes good for health?'\n"
            "✅ **Better Query:** 'Health benefits of tomatoes'\n\n"
            "❌ **Bad Query:** 'Can I eat rice before bed?'\n"
            "✅ **Better Query:** 'Effects of eating rice at night'\n\n"
            
            "📌 **Rule of Thumb:**\n"
            "If the original user query is **too broad, phrased as a question, or unlikely to match document content**, **restructure it** into a **topic-based** search.\n\n"

            "### 🔍 **Step 2: Retrieve Information Using `retrieve_from_doc`**\n"
            "Once the optimized query is ready:\n"
            "   - **Use the `retrieve_from_doc` tool** with the improved query.\n"
            "   - Ensure you **only call this tool once per request** (check conversation history).\n"
            "   - If the document search returns `['NO RESULT!']`, proceed directly to Step 3.\n\n"

            "### 🔍 **Step 3: Clean the Retrieved Data Using `ToFilter`**\n"
            "After retrieving content:\n"
            "   - **Call `ToFilter`** to refine the retrieved data.\n"
            "   - Provide:\n"
            "     - `retrieved_content`: The raw text from `retrieve_from_doc`.\n"
            "     - `user_query`: The original user question to guide filtering.\n"
            "   - **DO NOT modify or analyze the content yourself.**\n\n"

            "### 🚨 **Strict Rules to Follow:**\n"
            "✔ **NEVER respond directly to the user**—your job is tool invocation only.\n"
            "✔ **DO NOT assume an answer**—your role is to find relevant information, not interpret it.\n"
            "✔ **DO NOT perform multiple searches**—use `retrieve_from_doc` **only once per request**.\n"
            "✔ **DO NOT modify `retrieved_content`**—pass it directly to `ToFilter` without changes.\n"
            "✔ **IF NO RESULTS FOUND**, still call `ToFilter` with `retrieved_content = 'NO RESULT!'`.\n\n"

            "📢 **Final Reminder:**\n"
            "Your effectiveness depends on how well you **optimize the search query** to retrieve useful information. "
            "Always ensure **relevant, well-structured document retrieval** while maintaining strict tool invocation accuracy.\n"
        ),
        ("placeholder", "{messages}"),
    ]
)





doc_retrieval_safe_tools = [retrieve_from_doc]
doc_retrieval_sensitive_tools = []
doc_retrieval_tools = doc_retrieval_safe_tools + doc_retrieval_sensitive_tools
doc_retrieval_runnable = doc_retrieval_prompt | llm.bind_tools(
    doc_retrieval_tools + [ToFilter], tool_choice="any"
)