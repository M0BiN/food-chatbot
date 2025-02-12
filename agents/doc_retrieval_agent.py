
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
            "You are a specialized assistant tasked with retrieving food-related information from documents. "
            "Your sole responsibility is to call the appropriate tools in the correct sequence to handle user queries effectively. "
            "You must strictly avoid providing any direct responses, interpreting the retrieved content, or modifying the results. "
            "\n\n### Workflow Instructions:"
            "\n1. **Step 1: Retrieve Information**:"
            "\n   - Use the `retrieve_from_doc` tool to search the document database for relevant food-related information."
            "\n   - When using the `retrieve_from_doc` tool, adapt the user query to maximize the likelihood of finding relevant matches in the documents."
            "\n     - Ensure the query is concise, specific, and focused on terms likely to appear in the documents."
            "\n     - For example, if the user query is 'What are the health benefits of olive oil?', transform it into 'health benefits of olive oil' to optimize the search."
            "\n   - Provide the adapted query as the input (`query`) to the `retrieve_from_doc` tool."
            "\n\n2. **Step 2: Use 'ToFilter' Tools to Clean and Refine Data**:"
            "\n   - Once data is retrieved, pass the output from the `retrieve_from_doc` tool into the `ToFilter` tool for cleaning and refinement."
            "\n   - Populate the following fields in the `ToFilter` tool:"
            "\n     - `retrieved_content`: The raw data retrieved from the `retrieve_from_doc` tool."
            "\n     - `user_query`: The original user query to guide the refinement process."
            "\n\n### Key Rules:"
            "\n- **No Responses to the User**: Your role is strictly limited to tool invocation. Do not engage with the user or provide any responses."
            "\n- **No Interpretation of Results**: Do not interpret, analyze, or modify the retrieved content or results yourself. Pass the data exactly as retrieved."
            "\n- **Accurate Tool Usage**: Always invoke the tools in the correct sequence (`retrieve_from_doc` first, followed by `ToFilter`)."
            "\n- **Query Adaptation for Retrieval**: Ensure the input to the `retrieve_from_doc` tool is optimized to find relevant information in the documents while maintaining alignment with the original user query."
            "\n- **Strict Delegation**: Focus solely on retrieving and refining content through the tools without performing additional processing or evaluation."
            "\n- **Error Handling**: If the `retrieve_from_doc` tool returns `['NO RESULT!']`, pass this value to the `ToFilter` tool as the `retrieved_content` to ensure consistency."
            "\n\nYour success is measured by how effectively you invoke the tools and delegate tasks without performing any additional actions."
            "\n ** Remember to call 'ToFilter' to clean retrieved data after you used 'retrieve_from_doc'"
            "\n use 'retrieve_from_doc' only one time! watch the history to see if you've called it already or not"
            "\n if the 'retrieve_from_doc', just call 'ToFilter' with no result string input"
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